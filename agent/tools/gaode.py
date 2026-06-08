"""
高德工具集 — MCP stdio 方式接入
===============================
通过 ModelScope 上的 Gaode MCP Server 加载工具。
本地不写工具定义，全由 MCP Server 提供。
"""

import logging
from mcp.client.stdio import stdio_client

logger = logging.getLogger("agent.tools.gaode")


async def load_gaode_tools():
    """
    通过 stdio 连接 Gaode MCP Server，加载全部工具。
    需要 langchain-mcp-adapters:
        pip install langchain-mcp-adapters

    Gaode MCP Server 命令（根据实际包名调整）:
        npx @anthropic/mcp-server-gaode --apikey=xxx
        python -m gaode_mcp --apikey=xxx
    """
    from config import CFG

    key = CFG.get("gaode_api_key", "")
    if not key or key == "your_gaode_key_here":
        logger.warning("[GaodeMCP] API Key 未配置，跳过加载")
        return []

    # Gaode MCP Server 启动命令（按实际包名改）
    cmd = ["npx", "-y", "@modelcontextprotocol/server-gaode", "--apikey", key]

    try:
        from langchain_mcp_adapters.client import load_mcp_tools
        from mcp.client.session import ClientSession

        read, write = await stdio_client(cmd).__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()

        tools = await load_mcp_tools(session)
        logger.info(f"[GaodeMCP] 从 MCP 加载 {len(tools)} 个工具: {[t.name for t in tools]}")
        return tools

    except ImportError:
        logger.warning("[GaodeMCP] langchain_mcp_adapters 未安装，跳过加载")
        return []
    except Exception as e:
        logger.warning(f"[GaodeMCP] 连接失败: {e}，回退到本地工具")
        return _fallback_tools()


def _fallback_tools():
    """MCP 不可用时的本地回退工具"""
    import httpx
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field
    from config import CFG

    key = CFG.get("gaode_api_key", "")

    def _geocode(address: str) -> str:
        resp = httpx.get("https://restapi.amap.com/v3/geocode/geo",
                          params={"key": key, "address": address}, timeout=10)
        data = resp.json()
        if data["status"] != "1":
            return f"地址解析失败: {data.get('info', '')}"
        return f"{address} → {data['geocodes'][0]['location']}"

    def _search_around(location: str, keywords: str = "", radius: int = 2000) -> str:
        params = {"key": key, "location": location, "radius": radius, "extensions": "base", "offset": 5}
        if keywords:
            params["keywords"] = keywords
        resp = httpx.get("https://restapi.amap.com/v3/place/around", params=params, timeout=10)
        data = resp.json()
        if data["status"] != "1":
            return f"周边搜索失败: {data.get('info', '')}"
        pois = data.get("pois", []) or []
        lines = [f"找到 {len(pois)} 个结果:"] + [
            f"  - {p['name']}（{p.get('address','')}，{p.get('distance','?')}米）"
            for p in pois[:5]
        ]
        return "\n".join(lines)

    class GeoInput(BaseModel):
        address: str = Field(description="地址名称")

    class AroundInput(BaseModel):
        location: str = Field(description="经纬度")
        keywords: str = Field(default="", description="搜索关键词")
        radius: int = Field(default=2000, description="搜索半径米")

    return [
        StructuredTool.from_function(name="geocode", func=_geocode, args_schema=GeoInput,
            description="地址转经纬度。当用户提供了地址需要查询坐标时使用。"),
        StructuredTool.from_function(name="search_around", func=_search_around, args_schema=AroundInput,
            description="周边搜索。根据经纬度搜索附近场所（跑步路线、餐厅等）。"),
    ]
