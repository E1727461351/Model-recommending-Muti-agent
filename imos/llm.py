"""本地大模型结构化推理、原始响应保存与实际耗时统计。"""

import json
import time
import urllib.request
from pathlib import Path
from typing import Dict, List
from .io import write_json


class LLMClient:
    def __init__(self, config: Dict, model: str, output: Path, seed: int):
        self.config,self.model,self.output,self.seed=config,model,Path(output),seed
        self.calls: List[Dict]=[]

    def ask(self, role: str, payload: Dict, instruction: str, mock_answer: Dict) -> Dict:
        start=time.perf_counter()
        failure=None
        if self.config["llm_backend"]=="mock":
            answer=mock_answer
            raw={"message":{"content":json.dumps(answer,ensure_ascii=False)},"simulation":True}
        else:
            messages=[{"role":"user","content":json.dumps(payload,ensure_ascii=False)}]
            # 纯LLM基线仅给任务、规则与注册表，不添加系统规划提示
            if instruction:
                messages.insert(0,{"role":"system","content":instruction})
            options={k:v for k,v in self.config["llm"].items() if k not in ("timeout_s","keep_alive")}
            options["seed"]=self.seed
            body={"model":self.model,"messages":messages,"stream":False,"options":options,"keep_alive":self.config["llm"]["keep_alive"]}
            if instruction:
                body["format"]="json"
            request=urllib.request.Request(self.config["ollama_url"].rstrip("/")+"/api/chat",data=json.dumps(body).encode("utf-8"),headers={"Content-Type":"application/json"},method="POST")
            try:
                with urllib.request.urlopen(request,timeout=self.config["llm"]["timeout_s"]) as response:
                    raw=json.load(response)
            except (OSError,ValueError) as e:
                raw={"error":str(e)};failure=e
            answer=None
            if failure is None:
                try:
                    content=raw["message"]["content"].strip()
                    if content.startswith("```"):
                        content="\n".join(content.splitlines()[1:-1])
                    answer=json.loads(content)
                    if not isinstance(answer,dict):
                        raise ValueError("大模型输出必须为JSON对象")
                except (ValueError,KeyError,TypeError) as e:
                    failure=e
        timing={"stage":role,"model":self.model,"wall_s":time.perf_counter()-start,"options":dict(self.config["llm"],seed=self.seed),"backend":self.config["llm_backend"]}
        for k in ("total_duration","load_duration","prompt_eval_duration","eval_duration","prompt_eval_count","eval_count"):
            if k in raw:
                timing[k+"_s" if k.endswith("duration") else k]=raw[k]/1e9 if k.endswith("duration") else raw[k]
        self.calls.append(timing)
        write_json(self.output/f"llm_{len(self.calls):03d}_{role}.json",{"request":payload,"system":instruction,"response":raw,"timing":timing,"error":str(failure) if failure else None})
        if failure is not None:
            raise ValueError("本地大模型调用或结构化解析失败："+str(failure)) from failure
        return answer
