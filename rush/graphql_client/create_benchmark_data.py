# Generated by ariadne-codegen
# Source: gql

from .base_model import BaseModel
from .fragments import BenchmarkDataFields


class CreateBenchmarkData(BaseModel):
    create_benchmark_data: "CreateBenchmarkDataCreateBenchmarkData"


class CreateBenchmarkDataCreateBenchmarkData(BenchmarkDataFields):
    pass


CreateBenchmarkData.model_rebuild()
