from fastapi import APIRouter
from app.api.api_v1.endpoints.casual_inference import router as causal_inference_router

api_router = APIRouter()

@api_router.get("/health-check")
def health_check():
    return {"status": "OK"}


api_router.include_router(causal_inference_router, prefix="/causal", tags=["causal_inference"])