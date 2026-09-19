"""FastAPI production service for AnomaVision anomaly detection and similarity retrieval."""

from anomavision.api.app import create_app

__all__ = ["create_app"]
