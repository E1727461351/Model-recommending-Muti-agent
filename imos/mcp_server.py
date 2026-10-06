"""五类智能体的MCP工具服务，标准输出专用于协议通信。"""

import json
import sys
from pathlib import Path
import torch
from mcp.server.fastmcp import FastMCP
from .agents import AgentSystem, ExecutionFault
from .io import read_json


def main() -> None:
    config_path,model,output,seed=sys.argv[1:5]
    config=read_json(Path(config_path))
    torch.set_num_threads(config.get("torch_threads",1))
    system=AgentSystem(config,model,Path(output),int(seed))
    server=FastMCP("IMOS可信推荐智能体")

    def dispatch(stage: str, payload: str) -> str:
        try:
            return json.dumps(system.stage(stage,json.loads(payload)),ensure_ascii=False)
        except ExecutionFault as e:
            return json.dumps({"error":str(e),"cause":e.cause,"fault_type":e.fault_type},ensure_ascii=False)
        except (ValueError,RuntimeError,OSError,KeyError) as e:
            return json.dumps({"error":str(e)},ensure_ascii=False)

    @server.tool()
    def parse(payload: str) -> str:
        """检查指令一致性并解析任务。"""
        return dispatch("parse",payload)

    @server.tool()
    def plan(payload: str) -> str:
        """检索或生成标准动作序列。"""
        return dispatch("plan",payload)

    @server.tool()
    def match(payload: str) -> str:
        """将每个动作匹配到模型种类。"""
        return dispatch("match",payload)

    @server.tool()
    def schedule(payload: str) -> str:
        """执行可信过滤、图推理与实例精排。"""
        return dispatch("schedule",payload)

    @server.tool()
    def execute(payload: str) -> str:
        """在数据提供层加载输入并执行真实模型。"""
        return dispatch("execute",payload)

    @server.tool()
    def record_feedback(payload: str) -> str:
        """记录已完成根因归因的运行反馈。"""
        system.history.append(json.loads(payload))
        return json.dumps({"recorded":True})

    @server.tool()
    def save_logs() -> str:
        """保存当前会话完整执行轨迹和耗时。"""
        system.save()
        return json.dumps({"saved":True})

    server.run(transport="stdio")


if __name__=="__main__":
    main()
