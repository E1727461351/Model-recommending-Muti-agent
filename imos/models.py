"""候选模型训练、真实推理、权重扰动及验证性能记录。"""

import copy
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from .data import load_dataset, prepare_signal, statistical_features
from .io import read_json, write_json, seed_all
from .metrics import performance, quality
from .networks import SignalNetwork
from .trust import safe_path, seal


FAMILIES = {"D1":["SVR","LSBoost"], "D2":["CNN","BiTCN","CBTA"],
            "D3":["1D-CNN","LSTM","AttentionLSTM"], "D4":["TCN-Bi-LSTM","CNN-Bi-LSTM"], "Dr":["CNN","RNN"]}


def skeleton() -> List[Dict]:
    result=[]
    for dataset,names in FAMILIES.items():
        task="classification" if dataset in ("D3","Dr") else "regression"
        for name in names:
            for variant in range(3):
                ident = dataset+"/"+name+("-v"+str(variant) if variant else "")
                result.append({"id":ident,"name":ident.split("/")[-1],"architecture":name,"variant":variant,
                               "dataset":dataset,"task":task,"kind":"分类器类" if task=="classification" else "数值回归类",
                               "source":None,"publisher":None,"signature":None,"version":"1.0.0",
                               "artifacts":[],"baseline_hashes":{},"validation":{},"environment":{},
                               "status":"awaiting_artifact", "API":"imos.models.predict"})
    return result


def train_neural(x: np.ndarray,y: np.ndarray,xv: np.ndarray,yv: np.ndarray,model: Dict,config: Dict,device: str) -> Dict:
    mean=x.mean(axis=tuple(range(x.ndim-1)),keepdims=True)
    std=x.std(axis=tuple(range(x.ndim-1)),keepdims=True).clip(min=1e-6)
    xn,xvn=(x-mean)/std,(xv-mean)/std
    classes = np.unique(y).tolist() if model["task"]=="classification" else []
    if classes:
        mapping={v:i for i,v in enumerate(classes)}
        try:
            yt=np.array([mapping[v] for v in y.reshape(-1)])
            yvt=np.array([mapping[v] for v in yv.reshape(-1)])
        except KeyError as e:
            raise ValueError("验证集中出现训练集未见类别") from e
        y_mean,y_std=0.0,1.0
    else:
        y_mean,y_std=float(np.mean(y)),max(float(np.std(y)),1e-6)
        yt,yvt=(y.reshape(-1)-y_mean)/y_std,(yv.reshape(-1)-y_mean)/y_std
    channels=x.shape[-1] if x.ndim==3 else 1
    net=SignalNetwork(model["architecture"],channels,len(classes) or 1,config).to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    loss_fn=torch.nn.CrossEntropyLoss() if classes else torch.nn.MSELoss()
    best,state,stale=float("inf"),None,0
    history=[]
    rng=np.random.default_rng(config.get("seed",0))
    for epoch in range(config["epochs"]):
        net.train(); losses=[]
        indices=rng.permutation(len(x))
        for start in range(0,len(x),config["batch_size"]):
            batch=indices[start:start+config["batch_size"]]
            xb=torch.tensor(xn[batch],dtype=torch.float32,device=device)
            target=torch.tensor(yt[batch],dtype=torch.long if classes else torch.float32,device=device)
            optimizer.zero_grad(); pred=net(xb)
            loss=loss_fn(pred if classes else pred[:,0],target)
            loss.backward(); optimizer.step();losses.append(float(loss.detach().cpu()))
        net.eval()
        with torch.no_grad():
            vp=net(torch.tensor(xvn,dtype=torch.float32,device=device))
            target=torch.tensor(yvt,dtype=torch.long if classes else torch.float32,device=device)
            vl=float(loss_fn(vp if classes else vp[:,0],target).cpu())
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"val_loss":vl})
        if vl<best:
            best,state,stale=vl,copy.deepcopy(net.cpu().state_dict()),0
            net.to(device)
        else:
            stale+=1
            if stale>=config["patience"]:
                break
    return {"state_dict":state,"architecture":model["architecture"],"config":config,"channels":channels,
            "classes":classes,"mean":torch.tensor(mean),"std":torch.tensor(std),"y_mean":y_mean,"y_std":y_std,"training":history}


def predict(model: Dict, root: Path, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    artifact=safe_path(root,model["artifact"])
    if artifact.suffix==".pkl":
        # 仅允许在来源和完整性校验之后加载本地可信制品
        with artifact.open("rb") as stream:
            estimator=pickle.load(stream)
        return estimator.predict(statistical_features(x))
    payload=torch.load(artifact,map_location="cpu",weights_only=True)
    net=SignalNetwork(payload["architecture"],payload["channels"],len(payload["classes"]) or 1,payload["config"])
    net.load_state_dict(payload["state_dict"]);net.to(device).eval()
    mean,std=payload["mean"].numpy(),payload["std"].numpy()
    preds=[]
    with torch.no_grad():
        for start in range(0,len(x),payload["config"]["batch_size"]):
            batch=torch.tensor((x[start:start+payload["config"]["batch_size"]]-mean)/std,dtype=torch.float32,device=device)
            p=net(batch).cpu().numpy()
            preds.append(np.asarray(payload["classes"])[p.argmax(1)] if payload["classes"] else p[:,0]*payload["y_std"]+payload["y_mean"])
    return np.concatenate(preds)


def train_models(config: Dict, key_file: Path, publisher: str, source: str) -> List[Dict]:
    seed_all(config["seed"])
    root=Path(config["model_root"]);root.mkdir(parents=True,exist_ok=True)
    records=skeleton(); cfg=dict(config["base_training"],seed=config["seed"])
    data_cache={}
    for dataset in FAMILIES:
        scene="D2-a" if dataset=="D2" else dataset
        data=load_dataset(Path(config["dataset_root"]),scene)
        for split in ("train","val","test"):
            data["X_"+split]=prepare_signal(data["X_"+split],config["preprocess"])
        data_cache[dataset]=data
    for model in records:
        data=data_cache[model["dataset"]]
        x,y,xv,yv=data["X_train"],data["y_train"].reshape(-1),data["X_val"],data["y_val"].reshape(-1)
        path=root/(model["id"]+(".pkl" if model["architecture"] in ("SVR","LSBoost") else ".pt"))
        path.parent.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter()
        if model["architecture"] in ("SVR","LSBoost"):
            # 非神经模型没有可直接扰动的网络权重，干扰项通过超参数退化生成
            factor=1/(1+model["variant"]*2)
            est=SVR(C=cfg["svr_C"]*factor,epsilon=cfg["svr_epsilon"]*(1+model["variant"])) if model["architecture"]=="SVR" else GradientBoostingRegressor(n_estimators=max(1,int(cfg["lsboost_estimators"]*factor)),learning_rate=cfg["lsboost_learning_rate"],max_depth=cfg["lsboost_max_depth"],random_state=config["seed"])
            pipeline=make_pipeline(StandardScaler(),est);pipeline.fit(statistical_features(x),y)
            with path.open("wb") as stream:
                pickle.dump(pipeline,stream)
        elif model["variant"]:
            base=root/(model["dataset"]+"/"+model["architecture"]+".pt")
            payload=torch.load(base,map_location="cpu",weights_only=True)
            # 扰动只作用于部分参数张量，保留可执行结构；不强制干扰模型一定变差
            gen=torch.Generator().manual_seed(config["seed"]+model["variant"])
            for index,(name,value) in enumerate(payload["state_dict"].items()):
                if index%2==0 and torch.is_floating_point(value):
                    noise=torch.randn(value.shape,generator=gen)*max(float(value.std()) if value.numel()>1 else 0.0,1e-3)*cfg["noise_ratios"][model["variant"]-1]
                    value.add_(noise)
            torch.save(payload,path)
        else:
            payload=train_neural(x,y,xv,yv,model,cfg,config["device"])
            torch.save(payload,path)
            write_json(path.with_suffix(".training.json"),payload["training"])
        model.update({"artifact":str(path.relative_to(root)).replace("\\","/"),"artifacts":[str(path.relative_to(root)).replace("\\","/")],
                      "config":cfg,"preprocess":config["preprocess"],"dependencies":{"torch":">=2.0,<3","scikit-learn":">=1.2,<2"},
                      "source":source,"publisher":publisher,"status":"trained","train_s":time.perf_counter()-start,
                      "input_dim":list(x.shape[1:]),"output_dim":[len(np.unique(y)) if model["task"]=="classification" else 1],"DataType":"时序信号" if x.ndim==3 else "数值特征"})
        for scene,spec in config["scenarios"].items():
            if ("D2" if scene.startswith("D2") else scene)!=model["dataset"]:
                continue
            validation=data if scene==("D2-a" if model["dataset"]=="D2" else model["dataset"]) else load_dataset(Path(config["dataset_root"]),scene)
            vx=validation["X_val"] if validation is data else prepare_signal(validation["X_val"],config["preprocess"])
            model["validation"][scene]=performance(validation["y_val"],predict(model,root,vx,config["device"]),model["task"])
        sample=xv[:1]
        for _ in range(cfg["warmup_runs"]):
            predict(model,root,sample,config["device"])
        times=[]
        for _ in range(cfg["profile_runs"]):
            if config["device"].startswith("cuda"):
                torch.cuda.synchronize()
            t=time.perf_counter();predict(model,root,sample,config["device"])
            if config["device"].startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter()-t)*1000)
        # 加载+单样本推理耗时；论文未提供原模型的资源阈值，D1资源需求为可配置重建假设
        ram = 3.0 if model["id"].startswith("D1/LSBoost") else 0.1
        model["environment"]={"gpu_memory_gb":0,"ram_gb":ram,"cpu_cores":1,"storage_gb":path.stat().st_size/1e9,
                               "latency_ms":float(np.mean(times)),"throughput":1000/max(float(np.mean(times)),1e-8),"availability":100,
                               "os":"*","runtime":"*","runtime_version":"*","auth_level":0,"privacy_level":0}
        model["profile"]={"load_and_predict_ms":times,"resource_origin":"RAM是复现配置要求，不是实际峰值内存测量"}
        metadata_path=path.with_suffix(".config.json")
        write_json(metadata_path,{"architecture":model["architecture"],"training":cfg,"preprocess":config["preprocess"],"dependencies":model["dependencies"]})
        model["artifacts"].append(str(metadata_path.relative_to(root)).replace("\\","/"))
        seal(model,root,key_file)
    write_json(root/"registry.json",records)
    return records


def selector_records(config: Dict, models: List[Dict]) -> List[Dict]:
    from .graph import context_features
    from .trust import trust_values
    for model in models:
        src,intact,_=trust_values(model,Path(config["model_root"]),config,[])
        if src<config["source_threshold"] or intact!=1:
            raise ValueError("图训练不能加载未验证制品："+model["id"])
    result=[]
    for scene,spec in config["scenarios"].items():
        data=load_dataset(Path(config["dataset_root"]),scene)
        x=prepare_signal(data["X_val"],config["preprocess"]);y=data["y_val"]
        pool=[m for m in models if m["dataset"]==("D2" if scene.startswith("D2") else scene)]
        if len(x)<4:
            raise ValueError("图模型训练要求每场景至少4条验证数据")
        # 上下文按验证集分块；分块不能恢复原论文的历史交互标签，需要README披露
        for i,indices in enumerate(np.array_split(np.arange(len(x)),min(4,len(x)))):
            labels={m["id"]:quality(performance(y[indices],predict(m,Path(config["model_root"]),x[indices],config["device"]),spec["task"]),spec["task"]) for m in pool}
            result.append({"context_id":scene+"-val-block-"+str(i),"scene":scene,
                           "features":context_features(x[indices],config["environment"],spec),"eligible":[m["id"] for m in pool],"labels":labels})
    return result
