from pydantic import BaseModel


class PredictResponse(BaseModel):
    class_name: str
    confidence: float
    top_3: list[dict]


class GradCAMResponse(BaseModel):
    class_name: str
    confidence: float
    heatmap_base64: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int
