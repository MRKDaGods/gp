from typing import Optional

from fastapi import APIRouter, Depends

from backend.config import CITYFLOW_DIR
from backend.dependencies import get_app_state
from backend.services.video_service import _extract_camera_id
from backend.state import AppState

router = APIRouter()


@router.get("/api/locations/governorates")
async def get_governorates():
    """Get Egypt governorates"""
    return {
        "success": True,
        "data": [
            {"id": "cairo", "name": "Cairo", "nameAr": "القاهرة"},
            {"id": "giza", "name": "Giza", "nameAr": "الجيزة"},
            {"id": "alexandria", "name": "Alexandria", "nameAr": "الإسكندرية"},
            {"id": "qalyubia", "name": "Qalyubia", "nameAr": "القليوبية"},
        ],
    }


@router.get("/api/locations/cities/{governorate_id}")
async def get_cities(governorate_id: str):
    """Get cities in governorate"""
    cities_map = {
        "cairo": [
            {"id": "downtown", "name": "Downtown", "nameAr": "وسط البلد"},
            {"id": "nasr_city", "name": "Nasr City", "nameAr": "مدينة نصر"},
            {"id": "heliopolis", "name": "Heliopolis", "nameAr": "مصر الجديدة"},
            {"id": "maadi", "name": "Maadi", "nameAr": "المعادي"},
        ],
        "giza": [
            {"id": "dokki", "name": "Dokki", "nameAr": "الدقي"},
            {"id": "mohandessin", "name": "Mohandessin", "nameAr": "المهندسين"},
            {"id": "6october", "name": "6th October", "nameAr": "٦ أكتوبر"},
        ],
        "alexandria": [
            {"id": "montaza", "name": "Montaza", "nameAr": "المنتزه"},
            {"id": "sidi_gaber", "name": "Sidi Gaber", "nameAr": "سيدي جابر"},
        ],
    }
    return {
        "success": True,
        "data": cities_map.get(governorate_id, cities_map["cairo"]),
    }


@router.get("/api/locations/zones/{city_id}")
async def get_zones(city_id: str):
    """Get zones in city"""
    zones_map = {
        "downtown": [
            {"id": "tahrir", "name": "Tahrir Square", "nameAr": "ميدان التحرير"},
            {"id": "ramses", "name": "Ramses", "nameAr": "رمسيس"},
            {"id": "ataba", "name": "Ataba", "nameAr": "العتبة"},
        ],
        "nasr_city": [
            {"id": "abbas", "name": "Abbas El Akkad", "nameAr": "عباس العقاد"},
            {"id": "makram", "name": "Makram Ebeid", "nameAr": "مكرم عبيد"},
        ],
    }
    return {
        "success": True,
        "data": zones_map.get(city_id, zones_map["downtown"]),
    }


@router.get("/api/cameras")
async def get_cameras(zoneId: Optional[str] = None, state: AppState = Depends(get_app_state)):
    """Get cameras discovered from current videos and cityflow directory."""
    camera_ids: set = set()

    for video in state.uploaded_videos.values():
        cam = _extract_camera_id(str(video.get("name", ""))) or _extract_camera_id(
            str(video.get("path", ""))
        )
        if cam:
            camera_ids.add(cam)

    if CITYFLOW_DIR.exists():
        for child in CITYFLOW_DIR.iterdir():
            if child.is_dir():
                cam = _extract_camera_id(child.name)
                if cam:
                    camera_ids.add(cam)

    data = [
        {
            "id": cam,
            "name": cam,
            "location": {"zone": zoneId or "cityflow", "source": "cityflowv2"},
        }
        for cam in sorted(camera_ids)
    ]

    return {"success": True, "data": data}
