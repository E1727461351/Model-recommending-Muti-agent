"""五类智能体、SGAP、Trust-GUMS及根因反馈闭环。"""

import copy
import re
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from .constraints import mismatch, constraint_fit
from .data import load_dataset, prepare_signal
from .graph import context_features, graph_predict
from .io import read_json, write_json
from .llm import LLMClient
from .metrics import performance, quality, similarity
from .models import predict
from .trust import trust_values
from .resources import snapshot


class ExecutionFault(RuntimeError):
    def __init__(self, cause: str, detail: str, fault_type: str = "other"):
        super().__init__(detail)
        self.cause,self.fault_type=cause,fault_type


def consistency(request: str) -> Dict:
    # 此确定性规则覆盖原文三个反例，开放语言仍由任务解析LLM参与核验
    models=set(re.findall(r"([A-Z])型",request))
    obj=int(not(len(models)>1 and any(w in request for w in ("同一","同时"))))
    con=int(not("本地" in request and "云端" in request and any(w in request for w in ("仅","禁止上传"))))
    op=int(not("振动数据" in request and any(w in request for w in ("禁止读取","禁止访问"))))
    return {"object":obj,"constraints":con,"operation":op,"accepted":bool(obj and con and op)}


class TaskParser:
    def __init__(self, llm: LLMClient, config: Dict):
        self.llm,self.config=llm,config

    def parse(self, scene: str, request: str) -> Dict:
        checks=consistency(request)
        spec=self.config["scenarios"][scene]
        expected={"category":spec["category"],"consistency":checks,"scene":scene}
        answer=self.llm.ask("parse",{"request":request,"scene":scene,"categories":list(dict.fromkeys(s["category"] for s in self.config["scenarios"].values()))},
                            "你是任务解析智能体。检查任务实体、约束、目标与操作一致性。输出JSON：category、consistency（object/constraints/operation为0或1，accepted为布尔值）、scene。拒绝自相矛盾的请求。",expected)
        flags=answer.get("consistency",{})
        if not checks["accepted"] or not flags.get("accepted") or any(flags.get(k)!=1 for k in ("object","constraints","operation")):
            return dict(answer,consistency=checks if not checks["accepted"] else flags,rejected=True)
        categories={s["category"] for s in self.config["scenarios"].values()}
        if answer.get("category") not in categories:
            raise ValueError("解析输出没有映射到已注册任务空间")
        return dict(answer,scene=scene,rejected=False)


class ActionPlanner:
    def __init__(self, llm: LLMClient, rules: Dict):
        self.llm,self.rules=llm,rules

    def validate(self, actions: List[str], task: str) -> None:
        if not actions or not all(a in self.rules["actions"] for a in actions):
            raise ValueError("动作没有对应的模型种类")
        current="原始数据"
        for action in actions:
            item=self.rules["actions"][action]
            if item["input"]!=current:
                raise ValueError(f"{action} 的上下游接口不一致")
            current=item["output"]
        target="类别" if task=="classification" else "预测值"
        if current!=target:
            raise ValueError("动作序列没有完成任务目标")

    def plan(self, parsed: Dict, task: str, use_sgap: bool = True, empty_rules: bool = False) -> List[str]:
        templates={} if empty_rules else self.rules["task_rules"]
        category=parsed["category"]
        if use_sgap and category in templates:
            actions=templates[category]
        else:
            partial=max(templates, key=lambda k:len(set(k.split("-")) & set(category.split("-"))),default=None)
            default=["数据预处理","特征提取","分类识别" if task=="classification" else "回归预测"]
            answer=self.llm.ask("plan",{"task":parsed,"reference":templates.get(partial),"actions":self.rules["actions"] if use_sgap else {}},
                                "你是制造过程规划专家。识别目标、拆解动作、分析依赖、验证完整性，仅输出JSON对象actions（动作名称有序列表），动作必须匹配给定模式。" if use_sgap else "根据任务输出JSON对象actions（动作名称有序列表）。",{"actions":default})
            actions=answer["actions"]
        self.validate(actions,task)
        if use_sgap:
            self.rules["task_rules"][category]=list(actions)
        return list(actions)


class KindMatcher:
    def __init__(self, rules: Dict, llm: LLMClient = None):
        self.rules,self.llm=rules,llm

    def match(self, actions: List[str]) -> List[Dict]:
        pairs=[{"action":a,**self.rules["actions"][a]} for a in actions]
        if self.llm:
            expected={"pairs":[{"action":p["action"],"kind":p["kind"]} for p in pairs]}
            answer=self.llm.ask("match",{"actions":actions,"action_rules":self.rules["actions"]},
                                "你是种类匹配智能体。结合动作的输入模态与输出目标，按顺序匹配规则中的模型种类，输出JSON：pairs，包含action与kind。",expected)
            if answer.get("pairs")!=expected["pairs"]:
                raise ValueError("模型种类映射没有与动作规则严格对齐")
        return pairs


class InstanceScheduler:
    def __init__(self, config: Dict, registry: List[Dict], llm: LLMClient, history: List[Dict]):
        self.config,self.registry,self.llm,self.history=config,registry,llm,history

    def schedule(self, scene: str, kind: str, environment: Dict, features: List[float], excluded: List[str], use_gums: bool = True, scope_all: bool = False) -> Dict:
        root=Path(self.config["model_root"])
        dataset="D2" if scene.startswith("D2") else scene
        pool=[m for m in self.registry if m["kind"]==kind and m["id"] not in excluded and (scope_all or m["dataset"]==dataset)]
        decisions=[];safe=[]
        for model in pool:
            src,intact,run=trust_values(model,root,self.config,self.history)
            why=mismatch(model["environment"],environment)
            rejected=(src<self.config["source_threshold"] or intact!=1 or bool(why)) if use_gums else False
            decisions.append({"model":model["id"],"Tsrc":src,"TI":intact,"Trun":run,"hard_mismatch":why,"rejected":rejected})
            if not rejected:
                safe.append(model)
        if not safe:
            raise ValueError("候选集合为空，可信度或环境硬约束不满足")
        ranked=[]
        if use_gums:
            context={"scene":scene,"features":features,"eligible":[m["id"] for m in safe]}
            u=graph_predict(root/"selector.pt",safe,context)
            for model,value in zip(safe,u):
                detail=next(r for r in decisions if r["model"]==model["id"])
                fit=constraint_fit(model["environment"],environment)
                hist=quality(model.get("validation",{}).get(scene,{}),model["task"])
                weights=self.config["utility_weights"]
                utility=weights[0]*value+weights[1]*fit+weights[2]*hist
                trust=self.config["beta"]*detail["Tsrc"]+(1-self.config["beta"])*detail["Trun"]
                score=self.config["gamma"]*utility+(1-self.config["gamma"])*trust
                ranked.append({"id":model["id"],"u":value,"ConstraintFit":fit,"HistoryPerf":hist,"Utility":utility,"Trust":trust,"Score":score})
            ranked.sort(key=lambda r:(-r["Score"],r["id"]))
            # 原文公式28要求argmax，LLM负责解释与核验，不能覆盖工具排序
            answer=self.llm.ask("schedule",{"kind":kind,"ranking":ranked,"environment":environment},"你是实例调度智能体。依据Trust-GUMS工具结果选择Score最高的实例，仅输出JSON：selected。",{"selected":ranked[0]["id"]})
            if answer.get("selected")!=ranked[0]["id"]:
                raise ValueError("LLM调度结果不满足Trust-GUMS的argmax约束")
        else:
            items=[{"id":m["id"],"input":m.get("input_dim"),"output":m.get("output_dim"),"metadata":m} for m in safe]
            answer=self.llm.ask("schedule",{"kind":kind,"models":items},"选择满足输入输出格式的模型，仅输出JSON：selected。",{"selected":safe[0]["id"]})
        selected=next((m for m in safe if m["id"]==answer.get("selected")),None)
        if selected is None:
            raise ValueError("LLM推荐了未注册实例")
        return {"selected":selected["id"],"ranking":ranked,"filter":decisions}


class ComputingExecutor:
    def __init__(self, config: Dict, registry: List[Dict], history: List[Dict], output: Path, llm: LLMClient = None):
        self.config,self.registry,self.history,self.output=config,registry,history,Path(output)
        self.llm=llm

    def execute(self, scene: str, selected: str, environment: Dict, amplitude: float = 1.0) -> Dict:
        model=next(m for m in self.registry if m["id"]==selected)
        root=Path(self.config["model_root"])
        src,intact,run=trust_values(model,root,self.config,self.history)
        if src<self.config["source_threshold"] or intact!=1:
            raise ExecutionFault("model","执行前再次校验来源或关键制品失败","fail")
        why=mismatch(model["environment"],environment)
        if why:
            raise ExecutionFault("resource","执行期可用环境不能满足实例约束："+str(why),"resource")
        data=load_dataset(Path(self.config["dataset_root"]),scene)
        raw=data["X_test"]*amplitude
        if not np.isfinite(raw).all():
            raise ExecutionFault("data","输入不完整或越界")
        input_bounds=self.config.get("monitor",{}).get("input_bounds")
        if input_bounds and (np.any(raw<input_bounds[0]) or np.any(raw>input_bounds[1])):
            raise ExecutionFault("data","输入超出用户配置的正常运行边界")
        x=prepare_signal(raw,self.config["preprocess"])
        start=time.perf_counter()
        before=snapshot()
        try:
            pred=predict(model,root,x,self.config["device"])
        except (MemoryError,torch_memory_error()) as e:
            raise ExecutionFault("resource",str(e),"resource") from e
        except (ValueError,RuntimeError,OSError) as e:
            raise ExecutionFault("model",str(e),"fail") from e
        elapsed=time.perf_counter()-start
        if not np.isfinite(pred).all():
            raise ExecutionFault("model","模型输出非有限值","output")
        output_bounds=self.config.get("monitor",{}).get("output_bounds")
        if output_bounds and (np.any(pred<output_bounds[0]) or np.any(pred>output_bounds[1])):
            raise ExecutionFault("model","输入与资源正常但模型输出超出配置边界","output")
        if elapsed*1000/max(len(x),1)>environment["max_latency_ms"]:
            raise ExecutionFault("model","单样本平均推理时延超过任务边界","timeout")
        if self.llm:
            monitored={"model":selected,"input_finite":True,"output_finite":True,"integrity":intact,"resource_constraints_satisfied":True,"elapsed_per_sample_ms":elapsed*1000/len(x)}
            judgement=self.llm.ask("execute",monitored,"你是计算执行智能体。依据提供的实际运行状态核验是否正常，不生成数值预测或访问原始数据，输出JSON：normal（布尔值）和cause（正常时null）。",{"normal":True,"cause":None})
            if judgement.get("normal") is not True:
                raise ExecutionFault("task","LLM执行状态核验未通过，需要重新规划")
        # 标签只进入独立评估器，调度提示和图上下文均不能访问测试标签
        perf=performance(data["y_test"],pred,self.config["scenarios"][scene]["task"])
        self.output.mkdir(parents=True,exist_ok=True)
        interval=None
        if model["task"]=="regression":
            validation_x=prepare_signal(data["X_val"],self.config["preprocess"])
            validation_pred=predict(model,root,validation_x,self.config["device"])
            residual=np.abs(data["y_val"].reshape(-1)-validation_pred)
            coverage=self.config.get("prediction_interval_coverage",0.95)
            quantile=min(1.0,np.ceil((len(residual)+1)*coverage)/len(residual))
            radius=float(np.quantile(residual,quantile,method="higher"))
            interval={"nominal_coverage":coverage,"half_width":radius,"origin":"重建补充：验证残差分位数预测区间；不是原文已说明的置信区间算法"}
            np.savez_compressed(self.output/"predictions.npz",y=data["y_test"],prediction=pred,lower=pred-radius,upper=pred+radius)
        else:
            np.savez_compressed(self.output/"predictions.npz",y=data["y_test"],prediction=pred)
        self.history.append({"time":time.time(),"model":selected,"model_fault":None,"cause":None})
        return {"success":True,"selected":selected,"performance":perf,"inference_s":elapsed,"sample_count":len(x),
                "prediction_interval":interval,
                "state":{"input_finite":True,"output_finite":True,"integrity":intact,"environment":environment,"amplitude":amplitude,"resources_before":before,"resources_after":snapshot()},
                "predictions_path":str(self.output/"predictions.npz")}


def torch_memory_error() -> type:
    import torch
    return torch.cuda.OutOfMemoryError


class AgentSystem:
    def __init__(self, config: Dict, model: str, output: Path, seed: int, registry: List[Dict] = None):
        self.config=copy.deepcopy(config);self.output=Path(output)
        registry_path=Path(config.get("registry_override",str(Path(config["model_root"])/"registry.json")))
        self.registry=copy.deepcopy(registry if registry is not None else read_json(registry_path))
        self.rules=read_json(Path("configs/rules.json"));self.history=[];self.trace=[]
        self.llm=LLMClient(config,model,output,seed)
        self.parser=TaskParser(self.llm,config);self.planner=ActionPlanner(self.llm,self.rules)
        self.matcher=KindMatcher(self.rules,self.llm)
        self.scheduler=InstanceScheduler(config,self.registry,self.llm,self.history)
        self.executor=ComputingExecutor(config,self.registry,self.history,output,self.llm)
        if config.get("unconstrained_execution"):
            self.executor.llm=None
            self.matcher.llm=None

    def stage(self, name: str, payload: Dict) -> Dict:
        start=time.perf_counter()
        if name=="parse":
            answer=self.parser.parse(**payload)
        elif name=="plan":
            answer={"actions":self.planner.plan(**payload)}
        elif name=="match":
            answer={"pairs":self.matcher.match(**payload)}
        elif name=="schedule":
            answer=self.scheduler.schedule(**payload)
        elif name=="execute":
            try:
                answer=self.executor.execute(**payload)
            except ExecutionFault as e:
                evidence={"selected":payload["selected"],"scene":payload["scene"],"environment":payload["environment"],"detected_cause":e.cause,"fault_type":e.fault_type,"detail":str(e),"previous_stages":self.trace[-2:]}
                interpretation=self.llm.ask("diagnose",evidence,"你是计算执行智能体。根据已采集的执行轨迹与异常证据说明根因。仅输出JSON：cause为data/model/resource/task之一，explanation为简短说明。不编造原始信号或结果。",{"cause":e.cause,"explanation":str(e)})
                self.trace.append({"stage":"diagnose","input":evidence,"output":interpretation,"deterministic_cause":e.cause})
                # 密码校验与资源事实优先，LLM解释不覆盖确定性安全检测的归因
                raise
        else:
            raise ValueError("未知智能体阶段")
        self.trace.append({"stage":name,"elapsed_s":time.perf_counter()-start,"input":payload,"output":answer})
        return answer

    def save(self) -> None:
        write_json(self.output/"trace.json",self.trace)
        write_json(self.output/"history.json",self.history)
        write_json(self.output/"llm_timing.json",self.llm.calls)
        write_json(self.output/"updated_rules.json",self.rules)
