"""候选模型—上下文二部图、双向消息传递及有用性联合训练。"""

from pathlib import Path
from typing import Dict, List, Tuple
import copy
import time
import numpy as np
import torch
from torch import nn
from .io import seed_all, write_json


MODEL_DIM = 14
CONTEXT_DIM = 14


def context_features(x: np.ndarray, environment: Dict, scene: Dict) -> List[float]:
    values = np.asarray(x,dtype=np.float64)
    # 仅数据提供层返回汇总统计，智能体提示不包含原始生产数据
    stats = [np.tanh(values.mean()), np.tanh(values.std()), np.tanh(np.sqrt((values*values).mean())),
             np.tanh(np.max(np.abs(values))), np.log1p(values.shape[1])/12,
             (values.shape[2] if values.ndim == 3 else 1)/16]
    resources = [environment["cpu_cores"]/32, environment["ram_gb"]/64,
                 environment["gpu_memory_gb"]/32, np.log1p(environment["max_latency_ms"])/12]
    return [float(v) for v in stats+resources+[float(scene["task"]=="classification"),float(scene.get("size_ratio",1)),float(scene.get("amplitude_ratio",1)),1.0]]


def model_features(model: Dict, scene: str) -> List[float]:
    env, p = model["environment"], model.get("validation",{}).get(scene,{})
    # 验证集性能和在线历史可用，测试集性能绝不进入节点特征
    accuracy = p.get("accuracy",p.get("R2",0))
    accuracy = accuracy if accuracy is not None else 0
    families = ("SVR","LSBoost","CNN","BiTCN","CBTA","LSTM","RNN")
    name = model["architecture"]
    bits = [float(f in name) for f in families]
    return [float(np.clip(accuracy,0,1)), np.log1p(env.get("latency_ms",0))/12,
            env.get("ram_gb",0)/64,env.get("gpu_memory_gb",0)/32,
            env.get("cpu_cores",1)/32,float(model.get("variant",0))/2,
            float(model["task"]=="classification")]+bits


def build_graph(models: List[Dict], contexts: List[Dict]) -> Dict[str, torch.Tensor]:
    if not models or not contexts:
        raise ValueError("二部图模型节点与上下文节点均不能为空")
    # 当前局部图含一个上下文节点；训练图可含多个历史上下文，边只连接同任务同场景的可行实例
    mf = torch.tensor([model_features(m,contexts[0]["scene"]) for m in models],dtype=torch.float32)
    cf = torch.tensor([c["features"] for c in contexts],dtype=torch.float32)
    adjacency = torch.zeros(len(models),len(contexts))
    for i,m in enumerate(models):
        for j,c in enumerate(contexts):
            if m["id"] in c["eligible"]:
                adjacency[i,j] = 1.0
    return {"models":mf,"contexts":cf,"adjacency":adjacency}


class BipartiteRegressor(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        h = config["hidden_dim"]
        self.model_project = nn.Linear(MODEL_DIM,h)
        self.context_project = nn.Linear(CONTEXT_DIM,h)
        self.model_layers = nn.ModuleList([nn.Linear(2*h,h) for _ in range(config["layers"])])
        self.context_layers = nn.ModuleList([nn.Linear(2*h,h) for _ in range(config["layers"])])
        self.dropout = nn.Dropout(config["dropout"])
        self.mlp = nn.Sequential(nn.Linear(2*h,h),nn.ReLU(),nn.Dropout(config["dropout"]),nn.Linear(h,1),nn.Sigmoid())

    def forward(self, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        m = torch.relu(self.model_project(graph["models"]))
        c = torch.relu(self.context_project(graph["contexts"]))
        a = graph["adjacency"]
        for ml,cl in zip(self.model_layers,self.context_layers):
            # 两个分区同步更新，模型节点间、上下文节点间均无直接边
            mn = a @ c / a.sum(1,keepdim=True).clamp(min=1)
            cn = a.T @ m / a.sum(0,keepdim=True).T.clamp(min=1)
            m,c = self.dropout(torch.relu(ml(torch.cat([m,mn],1)))), self.dropout(torch.relu(cl(torch.cat([c,cn],1))))
        pairs = torch.cat([m[:,None,:].expand(-1,len(c),-1),c[None,:,:].expand(len(m),-1,-1)],dim=-1)
        return self.mlp(pairs).squeeze(-1)


def fit_graph(records: List[Dict], models: List[Dict], config: Dict, checkpoint: Path, seed: int) -> Dict:
    started=time.perf_counter()
    seed_all(seed)
    if len(records) < 4:
        raise ValueError("至少需要4个历史上下文用于上下文级训练/验证划分")
    groups = sorted(set(r["context_id"] for r in records))
    order = np.random.default_rng(seed).permutation(len(groups))
    count = max(1,int(len(groups)*config["validation_fraction"]))
    held = {groups[i] for i in order[:count]}
    if len(held)==len(groups):
        raise ValueError("训练上下文为空")
    train = [r for r in records if r["context_id"] not in held]
    val = [r for r in records if r["context_id"] in held]
    net = BipartiteRegressor(config)
    optimizer = torch.optim.Adam(net.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    best, stale, state, log = float("inf"),0,None,[]

    def loss_for(batch: List[Dict]) -> torch.Tensor:
        # 每个记录为一次局部图；整个候选集合共享图消息而非逐对独立MLP
        losses = []
        for r in batch:
            pool = [m for m in models if m["id"] in r["eligible"]]
            graph = build_graph(pool,[r])
            pred = net(graph)[:,0]
            target = torch.tensor([r["labels"][m["id"]] for m in pool],dtype=torch.float32)
            losses.append(nn.functional.mse_loss(pred,target))
        return torch.stack(losses).mean()

    for epoch in range(config["epochs"]):
        net.train(); accumulated=[]
        for start in range(0,len(train),config["context_batch_size"]):
            optimizer.zero_grad()
            loss = loss_for(train[start:start+config["context_batch_size"]])
            loss.backward(); optimizer.step(); accumulated.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():
            validation = float(loss_for(val))
        log.append({"epoch":epoch+1,"train_mse":float(np.mean(accumulated)),"val_mse":validation})
        if validation < best:
            best,stale,state = validation,0,copy.deepcopy(net.state_dict())
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":state,"config":config,"model_dim":MODEL_DIM,"context_dim":CONTEXT_DIM},checkpoint)
    report={"training_context_ids":sorted(set(r["context_id"] for r in train)),"validation_context_ids":sorted(held),
            "best_validation_mse":best,"elapsed_s":time.perf_counter()-started,"epochs":log,"label_origin":"真实候选模型在独立验证集子块上的准确率或截断R2；不是论文已披露标签"}
    write_json(checkpoint.with_suffix(".training.json"),report)
    return report


def graph_predict(checkpoint: Path, models: List[Dict], context: Dict) -> List[float]:
    if not checkpoint.is_file():
        raise FileNotFoundError("缺少Trust-GUMS图模型权重，请先执行 train-selector")
    payload = torch.load(checkpoint,map_location="cpu",weights_only=True)
    net = BipartiteRegressor(payload["config"])
    net.load_state_dict(payload["state_dict"]); net.eval()
    with torch.no_grad():
        return net(build_graph(models,[context]))[:,0].tolist()
