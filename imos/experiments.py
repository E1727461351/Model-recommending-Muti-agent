"""论文表2至表8、消融图及案例的统一实验驱动。"""

import asyncio
import copy
import shutil
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
from .agents import AgentSystem, consistency
from .data import load_dataset, prepare_signal
from .io import read_json, write_json, write_csv, manifest
from .metrics import performance, quality, similarity, summarize
from .models import predict
from .pipeline import run_pipeline
from .trust import trust_values


def evaluate_catalog(config: Dict, registry: List[Dict]) -> Dict:
    catalog={}
    for scene,spec in config["scenarios"].items():
        dataset="D2" if scene.startswith("D2") else scene
        data=load_dataset(Path(config["dataset_root"]),scene)
        x=prepare_signal(data["X_test"],config["preprocess"])
        catalog[scene]={}
        for m in registry:
            if m["dataset"]==dataset:
                src,intact,_=trust_values(m,Path(config["model_root"]),config,[])
                if src<config["source_threshold"] or not intact:
                    raise ValueError(f"评估前模型可信校验失败：{m['id']}")
                catalog[scene][m["id"]]=performance(data["y_test"],predict(m,Path(config["model_root"]),x,config["device"]),spec["task"])
    return catalog


def enrich(trial: Dict, config: Dict, registry: List[Dict], catalog: Dict) -> Dict:
    scene=trial["scene"];spec=config["scenarios"][scene]
    pool=[m for m in registry if m["id"] in catalog[scene]]
    # 未给定最优实例时从验证集预定基准，不能利用测试标签替调度选模型
    optimal=spec.get("optimal") or max(pool,key=lambda m:quality(m["validation"].get(scene,{}),spec["task"]))["id"]
    trial["optimal"]=optimal
    trial["optimal_origin"]="论文预设" if spec.get("optimal") else "原文未给定，按验证集预定"
    selected=trial.get("selected")
    same=selected in catalog[scene]
    trial["same_scene"]=same
    trial["quality_gap"]=abs(quality(catalog[scene][selected],spec["task"])-quality(catalog[scene][optimal],spec["task"])) if same else 1.0
    trial["MS"]=similarity(trial["performance"],catalog[scene][optimal]) if trial.get("success") and spec["task"]=="regression" else None
    return trial


def plot_ablation(rows: List[Dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    llms=list(dict.fromkeys(r["LLM"] for r in rows));versions=["V1","V2","V3","V4"]
    lookup={(r["LLM"],r["version"]):r for r in rows}
    fig,ax=plt.subplots(figsize=(max(6,len(llms)*1.4),4.5));x=np.arange(len(llms))
    for i,v in enumerate(versions):
        ax.bar(x+(i-1.5)*0.18,[lookup[(l,v)]["SR_paper"] for l in llms],0.18,label=v,color=str(i/4))
    ax.set_xticks(x);ax.set_xticklabels([l.split(":")[0] for l in llms],rotation=20)
    ax.set_ylabel("SR (%)");ax.set_ylim(0,105);ax.legend();fig.tight_layout()
    fig.savefig(output/"figure3_ablation_SR.png",dpi=220);plt.close(fig)
    values=np.array([[lookup[(l,v)]["MS_plot"] for l in llms] for v in versions])
    fig,ax=plt.subplots(figsize=(max(6,len(llms)*1.3),4.5))
    display=np.minimum(values,0.4)
    img=ax.imshow(display,cmap="gray",vmin=0,vmax=0.4,aspect="auto")
    for i in range(len(versions)):
        for j in range(len(llms)):
            row=lookup[(llms[j],versions[i])]
            text=("failure" if not row["successful_N"] else "N/A") if row["MS"] is None else f"{values[i,j]:.4f}"
            ax.text(j,i,text,ha="center",va="center",color="white" if display[i,j]<0.18 else "black")
    ax.set_yticks(range(4));ax.set_yticklabels(versions);ax.set_xticks(range(len(llms)))
    ax.set_xticklabels([l.split(":")[0] for l in llms],rotation=20)
    ax.set_title("MS: color capped at 0.4; failures use 1.0 sentinel")
    fig.colorbar(img,ax=ax);fig.tight_layout();fig.savefig(output/"figure4_ablation_MS.png",dpi=220);plt.close(fig)


async def run_suite(config: Dict, only: str = "all") -> Dict:
    output=Path(config["output_root"]);output.mkdir(parents=True,exist_ok=True)
    registry=read_json(Path(config["model_root"])/"registry.json")
    write_json(output/"manifest.json",manifest(config,"synthetic_mock" if config["llm_backend"]=="mock" else "real"))
    # 真实实验必须完整提供数据、签名模型和图权重，不静默使用演示资产
    for scene in config["scenarios"]:
        load_dataset(Path(config["dataset_root"]),scene)
    if not (Path(config["model_root"])/"selector.pt").is_file():
        raise FileNotFoundError("图模型尚未训练")
    catalog=evaluate_catalog(config,registry);write_json(output/"test_catalog.json",catalog)
    trials=[];tables={};counter=0

    async def group(section: str, scene: str, llm: str, variant: str, count: int, run_config: Dict = None, fault: str = None) -> Dict:
        nonlocal counter
        rows=[]
        for repeat in range(count):
            counter+=1
            trial_dir=output/"trials"/section/scene/llm.replace(":","_")/variant/str(repeat)
            # 同场景、同LLM不同方法共享重复序号种子
            trial=await run_pipeline(run_config or config,llm,scene,trial_dir,config["seed"]+repeat,variant,fault)
            enrich(trial,config,registry,catalog)
            trial.update({"section":section,"repeat":repeat})
            write_json(trial_dir/"trial.json",trial);trials.append(trial);rows.append(trial)
        return summarize(rows)

    if only in ("all","rq1"):
        for name,scenes in (("table2",["D1","D2-a","D3","D4"]),("table3",["D2-a","D2-b","D2-c","Dr"])):
            rows=[]
            for llm in config["llms"]:
                for scene in scenes:
                    summary=await group(name,scene,llm,"full",config["repetitions"]["rq1"])
                    rows.append({"LLM":llm,"scene":scene,**summary})
            tables[name]=rows
            write_csv(output/(name+".csv"),rows)
            averages=[]
            for scene in scenes:
                subset=[r for r in rows if r["scene"]==scene]
                averages.append({"LLM":"Avg","scene":scene,**{key:float(np.mean([r[key] for r in subset])) for key in ("SR_paper","CR_paper_literal","SR_binary","elapsed_all_s")}})
            write_csv(output/(name+"_averages.csv"),averages)
    if only in ("all","rq2"):
        tables["table4"]=[{"model":name,**perf} for name,perf in catalog["D2-a"].items()]
        write_csv(output/"table4.csv",tables["table4"])
        rows=[]
        for variant in ("full","rule_matching","llm_only"):
            for llm in config["llms"]:
                rows.append({"Type":variant,"LLM":llm,**await group("table5","D2-a",llm,variant,config["repetitions"]["rq2"])})
        rows.append({"Type":"code_only","LLM":"-",**await group("table5","D2-a",config["llms"][0],"code_only",config["repetitions"]["rq2"])})
        tables["table5"]=rows;write_csv(output/"table5.csv",rows)
        # 注册表位置交换实验：只改变顺序，不预先设定大模型的选择结果
        probes=[]
        d2=[m for m in registry if m["dataset"]=="D2"]
        for llm in config["llms"]:
            if not llm.startswith(("llama","gemma")):
                continue
            for i in range(len(d2)):
                perm=copy.deepcopy(registry);first=next(j for j,m in enumerate(perm) if m["id"]==d2[0]["id"]);other=next(j for j,m in enumerate(perm) if m["id"]==d2[i]["id"])
                perm[first],perm[other]=perm[other],perm[first]
                trial=await run_pipeline(config,llm,"D2-a",output/"order_probe"/llm.replace(":","_")/str(i),config["seed"],"rule_matching",registry=perm)
                probes.append({"LLM":llm,"first_model":d2[i]["id"],"selected":trial.get("selected"),"success":trial["success"]})
        write_csv(output/"registry_order_probe.csv",probes)
    if only in ("all","rq3"):
        rows=[]
        llms=config["llms"] if config["rq3_llms"]=="all" else [config["rq3_llms"]]
        for llm in llms:
            for variant in ("V1","V2","V3","V4"):
                rows.append({"LLM":llm,"version":variant,**await group("ablation","D2-a",llm,variant,config["repetitions"]["rq3"])})
        tables["ablation"]=rows;write_csv(output/"ablation.csv",rows);plot_ablation(rows,output)
    if only in ("all","rq4"):
        llm=next((l for l in config["llms"] if l.startswith("deepseek")),config["llms"][0])
        requests=["同一待预测轴承同时指定为A型和B型，预测表面粗糙度", "所有生产数据仅允许本地处理，使用云端服务完成计算", "使用A型轴承实时振动数据预测粗糙度，但禁止读取该振动数据", "预测B型轴承磨削工序的表面粗糙度"]
        system=AgentSystem(config,llm,output/"instruction_checks",config["seed"])
        rows=[]
        for i,request in enumerate(requests):
            parsed=system.parser.parse("D2-a",request)
            rows.append({"instruction":i+1,"request":request,**parsed["consistency"],"rejected":parsed["rejected"]})
        system.save();tables["table6"]=rows;write_csv(output/"table6.csv",rows)
        isolated=output/"trust_fault_modelbase"
        if isolated.exists():
            raise FileExistsError("可信故障实验目录已存在；请更换输出目录避免覆盖旧实验")
        shutil.copytree(Path(config["model_root"]),isolated)
        trust_config=copy.deepcopy(config);trust_config["model_root"]=str(isolated)
        modified=read_json(isolated/"registry.json")
        for m in modified:
            if m["id"]=="D2/CBTA":
                m["publisher"]=None;m["source"]=None;m["version"]=None
            if m["id"]=="D2/CBTA-v1":
                with (isolated/m["artifact"]).open("ab") as stream:
                    stream.write(b"integrity_test")
        # 表7仅比较三个CBTA变体，不让其他完整模型改变实验候选集
        modified=[m for m in modified if m["id"].startswith("D2/CBTA")]
        write_json(isolated/"registry.json",modified)
        trust_trial=await run_pipeline(trust_config,llm,"D2-a",output/"trust_fault",config["seed"])
        rows=[]
        for m in modified:
            src,intact,run=trust_values(m,isolated,trust_config,[])
            rows.append({"Model":m["name"],"R2":catalog["D2-a"][m["id"]]["R2"],"TSRC":src,"TI":intact,"TRUN":run,"selected":trust_trial.get("selected")==m["id"],"completed":trust_trial["success"]})
        tables["table7"]=rows;write_csv(output/"table7.csv",rows)
        rows=[]
        for llm in config["llms"]:
            drift=await run_pipeline(config,llm,"Dr",output/"drift"/llm.replace(":","_"),config["seed"],fault="drift")
            resource=await run_pipeline(config,llm,"D1",output/"resource"/llm.replace(":","_"),config["seed"],fault="resource")
            write_json(output/"drift"/llm.replace(":","_")/"baseline_metrics.json",catalog["Dr"])
            detected=any(e.get("cause")=="resource" for e in resource["events"])
            reselected=detected and resource.get("selected")!="D1/LSBoost"
            error=(1-drift.get("performance",{}).get("accuracy",0)) if drift["success"] else None
            rows.append({"LLM":llm,"drift_reselected":any(e.get("action")=="ModelReschedule" for e in drift["events"]),"drift_completed":drift["success"],
                         "drift_error":error,"drift_quality_within_task":error<=config["dr_max_error"] if error is not None and config["dr_max_error"] is not None else None,
                         "resource_detected":detected,"resource_reselected":reselected,"resource_completed":resource["success"],"resource_final_model":resource.get("selected"),
                         "resource_fault_mode":config.get("resource_fault_mode","quota")})
        tables["table8"]=rows;write_csv(output/"table8.csv",rows)
    case=await run_pipeline(config,config["llms"][0],"D2-a",output/"case_study",config["seed"])
    write_json(output/"case_study.json",case)
    write_json(output/"all_trials.json",trials)
    write_json(output/"all_tables.json",tables)
    write_csv(output/"runtime.csv",[{"section":t["section"],"scene":t["scene"],"LLM":t["llm"],"variant":t["variant"],"repeat":t["repeat"],"wall_s":t["elapsed_s"],"inference_s":t.get("inference_s"),"success":t["success"]} for t in trials])
    lines=["# 实验输出", "", "模式："+("合成数据与脚本式推理，仅验证程序" if config["llm_backend"]=="mock" else "真实数据与本地大模型推理"),"", "原文数值只在 docs/paper_reference.json 中存档，不参与本次输出。", ""]
    for name,rows in tables.items():
        lines.extend([f"## {name}","",f"{len(rows)}行，详情见 {name}.csv。",""])
    (output/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    return {"output":str(output),"table_rows":{k:len(v) for k,v in tables.items()},"trial_count":len(trials),"mode":config["llm_backend"]}
