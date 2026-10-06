"""命令行入口：建库、转换、训练、实验、演示与资产审计。"""

import argparse
import asyncio
import base64
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from .data import build_cross_scenes, load_dataset
from .graph import fit_graph
from .io import read_json, write_json, write_csv, manifest
from .models import skeleton, train_models, selector_records
from .prepare import convert


def scaffold() -> None:
    for scene in ("D1","D2-a","D2-b","D2-c","D3","D4","Dr"):
        directory=Path("datasets")/scene;directory.mkdir(parents=True,exist_ok=True)
        (directory/".gitkeep").touch()
    root=Path("modelbase");root.mkdir(exist_ok=True)
    if not (root/"registry.json").exists():
        write_json(root/"registry.json",skeleton())
    for group in ("D1","D2","D3","D4","Dr"):
        directory=root/group;directory.mkdir(exist_ok=True);(directory/".gitkeep").touch()


def audit(config: Dict) -> Dict:
    missing=[]
    for scene in config["scenarios"]:
        try:
            load_dataset(Path(config["dataset_root"]),scene)
        except (FileNotFoundError,ValueError) as e:
            missing.append(str(e))
    registry_path=Path(config["model_root"])/"registry.json"
    registry=read_json(registry_path) if registry_path.is_file() else []
    from .trust import trust_values
    for model in registry:
        try:
            src,intact,_=trust_values(model,Path(config["model_root"]),config,[])
            if src<config["source_threshold"] or not intact:
                missing.append(model["id"]+" 尚无可验证的可信模型制品")
        except (ValueError,OSError) as e:
            missing.append(model["id"]+": "+str(e))
    if not (Path(config["model_root"])/"selector.pt").is_file():
        missing.append("Trust-GUMS图模型未训练")
    return {"ready":not missing,"models":len(registry),"missing":missing}


def demo(config: Dict, output: Path) -> Dict:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    from .experiments import run_suite
    if output.exists():
        raise FileExistsError("演示输出目录已存在，请使用新目录")
    output.mkdir(parents=True)
    cfg=copy.deepcopy(config)
    cfg.update({"dataset_root":str(output/"datasets"),"model_root":str(output/"modelbase"),"output_root":str(output/"results"),"llm_backend":"mock","device":"cpu","transport":"direct"})
    cfg["base_training"].update({"epochs":2,"hidden_dim":8,"batch_size":16,"patience":2,"lsboost_estimators":10,"profile_runs":2,"warmup_runs":1})
    cfg["graph_training"].update({"epochs":8,"hidden_dim":8,"patience":3})
    # 演示保留论文重复次数和全部五个名称，但没有调用任何真实LLM
    key=Ed25519PrivateKey.generate();key_path=output/"demo_private_key.pem"
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    pub=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
    cfg["public_keys"]={"演示发布者":base64.b64encode(pub).decode("ascii")}
    rng=np.random.default_rng(cfg["seed"])
    for scene in ("D1","D2-a","D3","D4","Dr"):
        values={}
        for split,n in (("train",32),("val",16),("test",16)):
            x=rng.normal(size=(n,32,3)).astype(np.float32)
            if cfg["scenarios"][scene]["task"]=="classification":
                y=(x[:,:,0].mean(1)>0).astype(np.int64)
            else:
                y=1+np.mean(x*x,axis=(1,2))+0.1*x[:,:,0].mean(1)
            values["X_"+split],values["y_"+split]=x,y
            values["groups_"+split]=np.array([split+str(i) for i in range(n)])
        directory=Path(cfg["dataset_root"])/scene;directory.mkdir(parents=True)
        np.savez_compressed(directory/"data.npz",**values)
    build_cross_scenes(Path(cfg["dataset_root"]),cfg)
    write_json(output/"demo_config.json",cfg)
    torch.set_num_threads(1)
    models=train_models(cfg,key_path,"演示发布者","synthetic-demo-local-only")
    records=selector_records(cfg,models)
    fit_graph(records,models,cfg["graph_training"],Path(cfg["model_root"])/"selector.pt",cfg["seed"])
    result=asyncio.run(run_suite(cfg))
    write_json(output/"verification.json",result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description="依据论文重建的可信计算模型推荐工程")
    parser.add_argument("--config",type=Path,default=Path("configs/experiment.json"))
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("scaffold",help="建立数据集目录和36个候选模型注册表项")
    commands.add_parser("audit",help="检查真实数据、模型制品、可信度和图模型是否齐全")
    commands.add_parser("build-cross",help="从D2-a构造D2-b和D2-c")
    cv=commands.add_parser("convert",help="按显式映射转换原始数据")
    cv.add_argument("mapping",type=Path);cv.add_argument("destination",type=Path)
    tr=commands.add_parser("train-models",help="训练候选模型与扰动项")
    tr.add_argument("--signing-key",type=Path,required=True);tr.add_argument("--publisher",required=True);tr.add_argument("--source",required=True)
    gr=commands.add_parser("train-selector",help="联合训练GNN与MLP")
    gr.add_argument("--interactions",type=Path)
    run=commands.add_parser("run",help="运行全部或单组论文实验")
    run.add_argument("--only",choices=("all","rq1","rq2","rq3","rq4"),default="all")
    run.add_argument("--output",type=Path);run.add_argument("--transport",choices=("direct","mcp"));run.add_argument("--backend",choices=("ollama","mock"))
    dm=commands.add_parser("demo",help="独立合成数据演示，不生成论文复现结果")
    dm.add_argument("--output",type=Path)
    args=parser.parse_args();cfg=read_json(args.config)
    if args.command=="scaffold":
        scaffold();print("已建立datasets、modelbase及注册表")
    elif args.command=="audit":
        result=audit(cfg);write_json(Path(cfg["output_root"])/"asset_audit.json",result)
        print("真实实验资产齐全" if result["ready"] else f"待补充 {len(result['missing'])} 项，见 {cfg['output_root']}/asset_audit.json")
    elif args.command=="convert":
        convert(args.mapping,args.destination)
    elif args.command=="build-cross":
        build_cross_scenes(Path(cfg["dataset_root"]),cfg)
    elif args.command=="train-models":
        result=train_models(cfg,args.signing_key,args.publisher,args.source)
        print(f"已训练并签名 {len(result)} 个候选实例")
    elif args.command=="train-selector":
        models=read_json(Path(cfg["model_root"])/"registry.json")
        records=read_json(args.interactions) if args.interactions else selector_records(cfg,models)
        write_json(Path(cfg["model_root"])/"selector_interactions.json",records)
        result=fit_graph(records,models,cfg["graph_training"],Path(cfg["model_root"])/"selector.pt",cfg["seed"])
        print(f"图训练完成，最佳验证MSE={result['best_validation_mse']:.6f}")
    elif args.command=="run":
        from .experiments import run_suite
        if args.output:
            cfg["output_root"]=str(args.output)
        if args.transport:
            cfg["transport"]=args.transport
        if args.backend:
            cfg["llm_backend"]=args.backend
        output=Path(cfg["output_root"])
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("输出目录不为空，请用 --output 指定新的实验目录")
        result=asyncio.run(run_suite(cfg,args.only));print(result)
    elif args.command=="demo":
        path=args.output or Path("outputs")/("demo_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
        print(demo(cfg,path))


if __name__=="__main__":
    main()
