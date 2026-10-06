"""使用已有合成演示制品验证真实MCP与LangChain通信，不冒充真实大模型实验。"""

import argparse
import asyncio
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch
from imos.io import read_json, write_json
from imos.pipeline import run_pipeline


class LocalFixture(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        request=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages=request["messages"]
        prompt=messages[0]["content"] if messages[0]["role"]=="system" else ""
        payload=json.loads(messages[-1]["content"])
        if "任务解析" in prompt:
            scene=payload["scene"]
            category="质量预测-B型轴承-表面粗糙度" if scene.startswith("D2") else "质量预测-B型轴承-刀具磨损"
            answer={"category":category,"scene":scene,"consistency":{"object":1,"constraints":1,"operation":1,"accepted":True}}
        elif "种类匹配" in prompt:
            answer={"pairs":[{"action":a,"kind":payload["action_rules"][a]["kind"]} for a in payload["actions"]]}
        elif "实例调度" in prompt:
            answer={"selected":payload["ranking"][0]["id"]}
        elif "根因" in prompt:
            answer={"cause":payload["detected_cause"],"explanation":"依据提供的确定性检测证据"}
        elif "运行状态" in prompt:
            answer={"normal":True,"cause":None}
        else:
            answer={"actions":["数据预处理","特征提取","回归预测"]}
        response={"message":{"content":json.dumps(answer,ensure_ascii=False)},"total_duration":1000000,"load_duration":0,"prompt_eval_duration":200000,"eval_duration":800000,"prompt_eval_count":10,"eval_count":10,"fixture":True}
        raw=json.dumps(response).encode("utf-8")
        self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(raw)));self.end_headers();self.wfile.write(raw)

    def log_message(self,format: str,*args) -> None:
        # 测试服务不打印HTTP请求，避免淹没MCP验证输出
        return


async def verify(config: dict, output: Path) -> dict:
    config=copy.deepcopy(config)
    config.update({"transport":"mcp","llm_backend":"ollama","inference_origin":"本地HTTP固定响应测试服务，不是真实LLM","device":"cpu"})
    server=ThreadingHTTPServer(("127.0.0.1",0),LocalFixture)
    config["ollama_url"]=f"http://127.0.0.1:{server.server_address[1]}"
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        first=await run_pipeline(config,config["llms"][0],"D2-a",output/"normal",config["seed"])
        resource=await run_pipeline(config,config["llms"][0],"D1",output/"resource",config["seed"],fault="resource")
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
    assert first["success"],first
    assert resource["success"] and resource["selected"]=="D1/SVR",resource
    assert any(e.get("cause")=="resource" for e in resource["events"])
    stages=[r["stage"] for r in read_json(output/"normal/trace.json")]
    assert all(s in stages for s in ("parse","plan","match","schedule","execute"))
    timings=read_json(output/"normal/llm_timing.json")
    assert len(timings)>=4
    result={"transport":"真实MCP标准输入输出","orchestration":"真实LangChain Runnable调用","llm":"本地HTTP测试固定响应，未调用真实大模型", "normal_completed":True,"resource_recovered":True,"final_resource_model":resource["selected"],"stages":stages,"llm_http_calls":len(timings)}
    write_json(output/"verification.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="验证真实MCP协议与LangChain，不运行真实LLM")
    parser.add_argument("demo_config",type=Path);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    print(asyncio.run(verify(read_json(args.demo_config),args.output)))
