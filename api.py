"""
api.py - FastAPI JSON API для прогнозирования стоимости недвижимости
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
import logging
from model_loader import get_predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API прогнозирования стоимости недвижимости",
    description="REST API на основе RandomForestRegressor (R²=0.595)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic-модели

class PropertyFeatures(BaseModel):
    size: float = Field(..., ge=20, le=1000, description="Площадь, м²")
    rooms_num: int = Field(..., ge=0, le=10, description="Количество комнат")
    building_age_num: float = Field(0, ge=0, le=100, description="Возраст здания, лет")
    total_floor_count_num: float = Field(10, ge=1, le=50, description="Всего этажей")
    sub_type: str = Field("Daire", description="Тип объекта")
    listing_type: int = Field(1, ge=1, le=2, description="Тип предложения: 1=продажа, 2=аренда")
    heating_type: str = Field("Kombi (Doğalgaz)", description="Отопление")
    city: str = Field("İstanbul", description="Город")
    floor_category: str = Field("floor_3", description="Категория этажа")

    class Config:
        json_schema_extra = {
            "example": {
                "size": 120,
                "rooms_num": 3,
                "building_age_num": 5,
                "total_floor_count_num": 10,
                "sub_type": "Daire",
                "listing_type": 1,
                "heating_type": "Kombi (Doğalgaz)",
                "city": "İstanbul",
                "floor_category": "floor_3",
            }
        }


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Прогнозируемая цена, TRY")
    min_price: float = Field(..., description="Нижняя граница (±MAE)")
    max_price: float = Field(..., description="Верхняя граница (±MAE)")
    confidence_interval: str = Field(..., description="Интервал ±MAE")
    mae: float = Field(..., description="MAE модели")
    r2_score: float = Field(..., description="R² модели")
    mape: float = Field(..., description="MAPE модели, %")


class ModelInfoResponse(BaseModel):
    status: str
    model_type: Optional[str] = None
    features_count: Optional[int] = None
    r2_score: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None


class MetadataResponse(BaseModel):
    cities: List[str]
    sub_types: List[str]
    heating_types: List[str]
    floor_categories: List[str]
    listing_types: List[int]


# Startup

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Запуск API...")
    predictor = get_predictor()
    if predictor.is_loaded:
        logger.info(f"✅ Модель {predictor.best_model_name} загружена")
        logger.info(f"   R²: {predictor.metrics.get('R²', 0):.3f}")
        logger.info(f"   MAE: {predictor.metrics.get('MAE', 0):,.0f} TRY")
    else:
        logger.error("❌ Модель не загружена")
        raise RuntimeError("Не удалось загрузить модель")


# Эндпоинты

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "API прогнозирования стоимости недвижимости",
        "status": "running",
        "version": "2.0.0",
    }


@app.get("/health", tags=["Health"])
async def health():
    p = get_predictor()
    return {
        "status": "healthy" if p.is_loaded else "unhealthy",
        "model_loaded": p.is_loaded,
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    p = get_predictor()
    info = p.get_model_info()
    return ModelInfoResponse(**info)


@app.get("/metadata", response_model=MetadataResponse, tags=["Info"])
async def metadata():
    """Справочники для формы ввода."""
    p = get_predictor()
    if not p.is_loaded:
        raise HTTPException(503, "Модель не загружена")
    return MetadataResponse(**p.metadata)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(features: PropertyFeatures):
    try:
        logger.info(f"Прогноз: size={features.size}, city={features.city}")

        p = get_predictor()
        if not p.is_loaded:
            raise HTTPException(503, "Модель не загружена")

        result = p.predict(features.dict())
        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(500, f"Ошибка прогноза: {str(e)}")




@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(features_list: List[PropertyFeatures]):
    p = get_predictor()
    if not p.is_loaded:
        raise HTTPException(503, "Модель не загружена")

    results = [p.predict(f.dict()) for f in features_list]
    return {"predictions": results, "count": len(results)}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")