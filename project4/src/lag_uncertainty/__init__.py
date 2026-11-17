"""Conditional autoregressive model comparison with explicit model uncertainty."""

from .data import ARDesignBuilder, IndexedARData
from .mixture import ModelAveragedInference
from .specification import PriorSpecification

__all__ = ["ARDesignBuilder", "IndexedARData", "ModelAveragedInference", "PriorSpecification"]
