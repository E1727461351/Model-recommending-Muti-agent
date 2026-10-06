"""控制平面到计算平面的端到端流程及有界恢复。"""

import copy
import time
from pathlib import Path
from typing import Dict, List
from .agents import AgentSystem, ExecutionFault
from .data import load_dataset
from .graph import context_features
from .io import write_json
from .transport import StageTransport
from .resources import PhysicalPressure, snapshot


async def run_pipeline(config: Dict, llm: str, scene: str, output: Path, seed: int,
                       variant: str = "full", fault: str = None, registry: List[Dict] = None) -> Dict:
    system=AgentSystem(config,llm,output,seed,registry)
    if variant in ("code_only","llm_only","V4"):
        system.config["unconstrained_execution"]=True
        system.executor.llm=None
        system.matcher.llm=None
    if registry is not None and config["transport"]=="mcp":
        # 可信注入实验使用独立注册表，服务端读取同一隔离副本
        isolated=Path(output)/"isolated_modelbase"
        isolated.mkdir(parents=True,exist_ok=True)
        # 模型文件不复制，保持原根目录；通过配置明确指定注册表文件
        write_json(Path(output)/"registry_override.json",registry)
        system.config["registry_override"]=str((Path(output)/"registry_override.json").resolve())
    start=time.perf_counter();spec=config["scenarios"][scene]
    request=spec["request"]+("；数据提供方报告振动幅值变为1.01倍，请依据任务要求继续计算" if fault=="drift" else "")
    environment=copy.deepcopy(config["environment"])
    data=load_dataset(Path(config["dataset_root"]),scene)
    # 节点统计由数据提供层生成，协调智能体只能获得汇总上下文
    features=context_features(data["X_test"]*(1.01 if fault=="drift" else 1.0),environment,spec)
    events=[];excluded=[];selected=None;retries=0;result={"success":False}
    use_sgap=variant not in ("V4","llm_only")
    use_gums=variant in ("full","V1","V2")
    pressure=None
    async with StageTransport(system) as transport:
        async def call(stage: str,payload: Dict) -> Dict:
            if config["llm_backend"]=="mock":
                return await transport.call(stage,payload)
            from langchain_core.runnables import RunnableLambda
            # LangChain负责阶段调用编排，MCP负责跨进程工具通信
            async def invoke(value: Dict) -> Dict:
                return await transport.call(stage,value)
            return await RunnableLambda(invoke).ainvoke(payload)

        for attempt in range(config["max_retries"]+1):
            try:
                if variant=="code_only":
                    parsed={"category":spec["category"],"scene":scene}
                    actions=system.rules["task_rules"][spec["category"]]
                    pairs=system.matcher.match(actions)
                    selected="D2/CBTA"
                elif variant in ("llm_only","V4"):
                    answer=system.llm.ask("unconstrained",{"request":request,"rules":system.rules,"registry":system.registry},"",{"category":spec["category"],"actions":["数据预处理","特征提取","分类识别" if spec["task"]=="classification" else "回归预测"],"selected":system.registry[0]["id"]})
                    parsed={"category":answer["category"],"scene":scene}
                    actions=answer["actions"];system.planner.validate(actions,spec["task"])
                    pairs=system.matcher.match(actions);selected=answer["selected"]
                else:
                    parsed=await call("parse",{"scene":scene,"request":request})
                    if parsed["rejected"]:
                        raise ValueError("指令被一致性检查拒绝")
                    actions=(await call("plan",{"parsed":parsed,"task":spec["task"],"use_sgap":use_sgap,"empty_rules":variant=="V2"}))["actions"]
                    pairs=(await call("match",{"actions":actions}))["pairs"]
                    final=next((p for p in reversed(pairs) if not p.get("default")),None)
                    if final is None:
                        raise ValueError("没有需要调度的终端计算动作")
                    decision=await call("schedule",{"scene":scene,"kind":final["kind"],"environment":environment,"features":features,"excluded":excluded,"use_gums":use_gums,"scope_all":variant=="V3"})
                    selected=decision["selected"]
                # 原文资源实验明确将LSBoost设置为默认模型，再在调度后改变资源
                if fault=="resource" and not events:
                    selected="D1/LSBoost"
                    if config.get("resource_fault_mode","quota")=="physical":
                        pressure=PhysicalPressure(config["resource_fault"]["cpu_fraction"],config["resource_fault"]["ram_gb"])
                        transport.stack.callback(pressure.__exit__,None,None,None)
                        pressure.__enter__()
                    environment["cpu_cores"]=max(1,int(environment["cpu_cores"]*config["resource_fault"]["cpu_fraction"]))
                    environment["gpu_memory_gb"]=config["resource_fault"]["gpu_memory_gb"]
                    environment["ram_gb"]=config["resource_fault"]["ram_gb"]
                    features=context_features(data["X_test"],environment,spec)
                    if pressure:
                        measured=snapshot()
                        environment["ram_gb"]=min(environment["ram_gb"],measured["available_ram_gb"])
                        environment["gpu_memory_gb"]=min(environment["gpu_memory_gb"],measured.get("available_gpu_gb",0))
                    events.append({"event":"resource_change_after_dispatch","model":selected,"environment":copy.deepcopy(environment),"mode":config.get("resource_fault_mode","quota")})
                events.append({"event":"bindings","task":parsed,"bindings":[dict(p,model=p.get("default") or selected) for p in pairs]})
                result=await call("execute",{"scene":scene,"selected":selected,"environment":environment,"amplitude":1.01 if fault=="drift" else 1.0})
                break
            except ExecutionFault as e:
                retries=attempt+1
                await transport.feedback(selected,e.cause,e.fault_type)
                events.append({"event":"detected","cause":e.cause,"model":selected,"detail":str(e),"action":{"data":"DataRecovery","model":"ModelReschedule","resource":"ResourceReschedule","task":"ActionReplan"}[e.cause]})
                if e.cause in ("resource","model"):
                    excluded.append(selected)
                if e.cause=="data":
                    # 重新读取数据源；若源没有恢复则有界失败，不编造替代输入
                    events.append({"event":"request_fresh_data"})
            except (ValueError,RuntimeError,OSError,KeyError,StopIteration) as e:
                retries=attempt+1
                events.append({"event":"planning_or_execution_error","detail":str(e),"action":"ActionReplan"})
        await transport.save()
    result.update({"scene":scene,"llm":llm,"variant":variant,"elapsed_s":time.perf_counter()-start,"retries":retries,"events":events,
                   "selected":result.get("selected",selected),"seed":seed,"mode":"synthetic_mock" if config["llm_backend"]=="mock" else "real"})
    write_json(Path(output)/"trial.json",result)
    return result
