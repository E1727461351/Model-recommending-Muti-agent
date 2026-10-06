"""本地直接调用与真实MCP标准输入输出通信适配。"""

import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Dict
from .agents import AgentSystem, ExecutionFault
from .io import write_json, read_json


class StageTransport:
    def __init__(self, system: AgentSystem):
        self.system=system;self.stack=AsyncExitStack();self.session=None

    async def __aenter__(self):
        if self.system.config["transport"]=="mcp":
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            config_path=self.system.output/"mcp_config.json"
            write_json(config_path,self.system.config)
            parameters=StdioServerParameters(command=sys.executable,args=["-m","imos.mcp_server",str(config_path.resolve()),self.system.llm.model,str(self.system.output.resolve()),str(self.system.llm.seed)],env=dict(os.environ))
            reader,writer=await self.stack.enter_async_context(stdio_client(parameters))
            self.session=await self.stack.enter_async_context(ClientSession(reader,writer))
            await self.session.initialize()
        return self

    async def __aexit__(self,exc_type,exc,tb):
        await self.stack.aclose()

    async def call(self, stage: str, payload: Dict) -> Dict:
        if self.session is None:
            return self.system.stage(stage,payload)
        result=await self.session.call_tool(stage,{"payload":json.dumps(payload,ensure_ascii=False)})
        if result.isError:
            raise RuntimeError(str(result.content))
        raw=json.loads(next(c.text for c in result.content if c.type=="text"))
        if raw.get("error"):
            if raw.get("cause"):
                raise ExecutionFault(raw["cause"],raw["error"],raw["fault_type"])
            raise ValueError(raw["error"])
        return raw

    async def feedback(self, model: str, cause: str, fault_type: str) -> None:
        import time
        entry={"time":time.time(),"model":model,"cause":cause,"model_fault":fault_type if cause=="model" else None}
        if self.session:
            await self.session.call_tool("record_feedback",{"payload":json.dumps(entry)})
        else:
            self.system.history.append(entry)

    async def save(self) -> None:
        if self.session:
            await self.session.call_tool("save_logs",{})
            if self.system.llm.calls:
                path=self.system.output/"llm_timing.json"
                remote=read_json(path) if path.is_file() else []
                write_json(path,remote+self.system.llm.calls)
        else:
            self.system.save()
