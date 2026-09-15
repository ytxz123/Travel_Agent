# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 路线规划工具 
# --------------------------------------------


import asyncio
import os
import httpx
from dotenv import load_dotenv

from utils.markdown import json2md
from utils.text import truncate_text
from qwen_agent.tools.base import BaseTool, register_tool

# 加载环境变量
load_dotenv()

AMAP_MAPS_API_KEY = os.environ["AMAP_MAPS_API_KEY"]


async def reverse_geocode(location: str):
    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {"key": AMAP_MAPS_API_KEY, "location": location}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
        finally:
            await client.aclose()

    return result


async def get_citycode(location: str):
    result = await reverse_geocode(location)

    try:
        citycode = result["regeocode"]["addressComponent"]["citycode"]
    except:
        citycode = None

    return citycode


async def driving_direction(
    origin: str, destination: str, waypoints: str | None = None
):
    url = "https://restapi.amap.com/v5/direction/driving?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    if waypoints:
        params["waypoints"] = waypoints

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
        finally:
            await client.aclose()

    return result


async def walking_direction(origin: str, destination: str):
    url = "https://restapi.amap.com/v5/direction/walking?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
        finally:
            await client.aclose()

    return result


async def bicycling_direction(origin: str, destination: str):
    url = "https://restapi.amap.com/v5/direction/bicycling?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
        finally:
            await client.aclose()

    return result


async def electrobike_direction(origin: str, destination: str):
    url = "https://restapi.amap.com/v5/direction/electrobike?parameters"
    params = {"key": AMAP_MAPS_API_KEY, "origin": origin, "destination": destination}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
        finally:
            await client.aclose()

    return result


async def transit_direction(origin: str, destination: str):
    url = "https://restapi.amap.com/v5/direction/transit/integrated?parameters"

    citycode_origin, citycode_destination = await asyncio.gather(
        get_citycode(origin), get_citycode(destination)
    )

    if not citycode_origin:
        return f"City not found for transit origin for {origin} and {destination}."

    if not citycode_destination:
        return f"City not found for transit destination for {origin} and {destination}."

    params = {
        "key": AMAP_MAPS_API_KEY,
        "origin": origin,
        "destination": destination,
        "city1": citycode_origin,
        "city2": citycode_destination,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
        finally:
            await client.aclose()

    return result


@register_tool("route_planning", allow_overwrite=True)
class RoutePlanning(BaseTool):
    name = "route_planning"
    description = "路线规划：驾车/步行/骑行/电动车/公交。"
    parameters = {
        "type": "object",
        "properties": {
            "origin": {
                "type": "string",
                "description": "起点经纬度，经度在前，格式 lng,lat"
            },
            "destination": {
                "type": "string",
                "description": "终点经纬度，经度在前，格式 lng,lat"
            },
            "mode": {
                "type": "string",
                "enum": ["driving", "walking", "bicycling", "electrobike", "transit"],
                "description": "路线类型，默认 driving"
            },
            "waypoints": {
                "type": "string",
                "description": "途经点，多个点以 ; 分隔，每点格式 lng,lat"
            },
        },
        "required": ["origin", "destination"]
    }


    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)


    async def call(self, params: dict, **kwargs) -> str:  # type: ignore[override]
        try:
            origin = params["origin"]
            destination = params["destination"]
            mode = params.get("mode", "driving")
            waypoints = params.get("waypoints")
        except Exception:
            return "[RoutePlanning] Invalid request format: Input must be a JSON object containing 'origin' and 'destination' field"

        if mode == "driving":
            result = await driving_direction(origin, destination, waypoints=waypoints)
        elif mode == "walking":
            result = await walking_direction(origin, destination)
        elif mode == "bicycling":
            result = await bicycling_direction(origin, destination)
        elif mode == "electrobike":
            result = await electrobike_direction(origin, destination)
        elif mode == "transit":
            result = await transit_direction(origin, destination)
        else:
            return f"[RoutePlanning] Unsupported mode: '{mode}'. Supported modes: driving, walking, bicycling, electrobike, transit"

        if result.get("status") != "1":
            msg = result.get("info", "unknown error")
            raise Exception(f"API response error: {msg}")

        route = result.get("route")
        if not route:
            raise Exception(f"No route available for {params}.")

        return truncate_text(json2md(route))


if __name__ == "__main__":
    search = RoutePlanning()
    params = {"origin": "104.06,30.67", "destination": "108.93,34.27", "mode": "driving", "waypoints": "105.84,32.44"}
    res = asyncio.run(search.call(params))
    print(res)
