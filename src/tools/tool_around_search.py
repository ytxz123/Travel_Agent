# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 周边搜工具 
# --------------------------------------------


import os
import asyncio
import httpx
from dotenv import load_dotenv

from utils.markdown import json2md
from utils.text import truncate_text
from qwen_agent.tools.base import BaseTool, register_tool

# 加载环境变量
load_dotenv()

AMAP_MAPS_API_KEY = os.environ["AMAP_MAPS_API_KEY"]


@register_tool("around_search", allow_overwrite=True)
class AroundSearch(BaseTool):
    name = "around_search"
    description = "以圆心+半径搜索周边地点。"
    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "中心点经纬度，格式 lng,lat"
            },
            "radius": {
                "type": "integer",
                "description": "半径（米），0-50000，默认5000"
            },
            "keyword": {
                "type": "string",
                "description": "可选，单个关键词"
            },
            "region": {
                "type": "string",
                "description": "可选，城市级区域（中文）"
            },
        },
        "required": ["location"]
    }


    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)


    async def call(self, params: dict, **kwargs) -> str:  # type: ignore[override]
        try:
            location = params["location"]
            radius = params.get("radius", 5000)
            keyword = params.get("keyword", None)
            region = params.get("region", None)
        except Exception:
            return "[AroundSearch] Invalid request format: Input must be a JSON object containing 'location' field"

        url = "https://restapi.amap.com/v5/place/around"
        params = {
            "key": AMAP_MAPS_API_KEY,
            "location": location,
            "radius": radius,
            "show_fields": "business",
        }
        if keyword:
            params["keywords"] = keyword
        if region:
            params["region"] = region

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                result = response.json()
            finally:
                await client.aclose()

        if result.get("status") != "1":
            msg = result.get("info", "unknown error")
            return f"API response error: {msg}"

        pois = result.get("pois")
        if not pois:
            return f"No POI data available for {params}."

        return truncate_text(json2md(pois))

if __name__ == "__main__":
    search = AroundSearch()
    params = {'location': '120.081964,30.302761', 'keyword': '商店'}
    res = asyncio.run(search.call(params))
    print(res)
