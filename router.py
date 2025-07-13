import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import mcp.types as types
import yaml
from dotenv import load_dotenv

from mcp_server_copilot.matcher import ToolMatcher
from mcp_server_copilot.mcp_connection import MCPConnection
from mcp_server_copilot.schemas import Server, ServerConfig

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def dump_to_yaml(data: dict[str, Any]) -> str:
    """将字典转换为YAML格式的字符串以便更好地显示。"""
    return yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


class Router:
    _default_config_path = PROJECT_ROOT / "config" / "config.sample.json"

    def __init__(
        self,
        config: dict[str, Any] | Path = _default_config_path,
    ):
        # self.connections 用于存储【已建立】的连接，初始为空
        self.connections = {}
        # self.servers 用于存储【所有已知】服务器的配置信息
        self.servers = {}

        if isinstance(config, dict):
            self.config = config
        elif isinstance(config, Path):
            if config.exists():
                with config.open("r") as f:
                    self.config = json.load(f)
            else:
                logger.warning(f"Config file not found at {config}. Starting with empty server list.")
                self.config = {"mcpServers": {}}
        else:
            raise ValueError("Config must be a dictionary or a Path to a JSON file.")
        
        # 解析配置，但不连接
        for name, config_data in self.config.get("mcpServers", {}).items():
            self.servers[name] = Server(name=name, config=ServerConfig(**config_data))
        
        load_dotenv()
        
        # 初始化 ToolMatcher
        self.matcher = ToolMatcher(
            embedding_model="text-embedding-v4",
            dimensions=1024,
            top_servers=5,
            top_tools=3
        )
        
        # 从环境变量中获取API密钥和数据路径
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        default_data_path = PROJECT_ROOT / "config" / "mcp_arg.json"
        data_path = os.getenv("MCP_DATA_PATH", default_data_path) # 根据您的日志，路径改为config下

        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable not set.")
        if not data_path or not os.path.exists(data_path):
             raise ValueError(f"MCP_DATA_PATH not set or file not found at: {data_path}")

        self.matcher.setup_openai_client(base_url=base_url, api_key=api_key)
        self.matcher.load_data(data_path)

    async def route(self, query: str) -> dict[str, Any]:
        """使用ToolMatcher进行路由，找到最匹配的工具。"""
        return self.matcher.match(query)

    # async def call_tool(
    #     self, server_name: str, tool_name: str, params: dict[str, Any] | None = None
    # ) -> types.CallToolResult:
    #     """在指定的服务器上执行工具，如果未连接则按需连接。"""
    #     connection = self.connections.get(server_name)

    #     # 如果连接不存在，则建立新连接
    #     if not connection:
    #         server_config = self.servers.get(server_name)
    #         if not server_config:
    #             raise ValueError(f"Server '{server_name}' is not defined in the configuration.")
            
    #         logger.info(f"Connection to server '{server_name}' not found, connecting on demand...")
    #         connection = MCPConnection(server_config)
    #         await connection.connect()
            
    #         # 将新建立的连接存储起来，以便复用
    #         self.connections[server_name] = connection

    #     # 现在可以确保连接已存在，执行工具调用
    #     return await connection.call_tool(tool_name, params or {})
            
    async def call_tool(
        self, server_name: str, tool_name: str, params: dict[str, Any] | None = None
    ) -> types.CallToolResult:
        """在指定的服务器上执行工具，如果未连接则按需连接。"""
        connection = self.connections.get(server_name)

        # 如果连接不存在，则建立新连接
        if not connection:
            server_config = self.servers.get(server_name)
            if not server_config:
                raise ValueError(f"Server '{server_name}' is not defined in the configuration.")
        
            connection = MCPConnection(server_config)

            await connection.connect()
            # 将新建立的连接存储起来，以便复用
            self.connections[server_name] = connection

        result = await connection.call_tool(tool_name, params or {})
        
        return result

    async def aclose(self):
        """关闭所有【已建立】的连接。"""
        await asyncio.gather(
            *[conn.aclose() for conn in self.connections.values()] 
        )

    async def __aenter__(self):
        """异步上下文管理器的进入方法。"""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """异步上下文管理器的退出方法。"""
        await self.aclose()