# Generated by ariadne-codegen
# Source: gql

from typing import Any

from .base_model import BaseModel


class DeleteBenchmark(BaseModel):
    delete_benchmark: "DeleteBenchmarkDeleteBenchmark"


class DeleteBenchmarkDeleteBenchmark(BaseModel):
    id: Any


DeleteBenchmark.model_rebuild()
