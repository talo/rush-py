#!/usr/bin/env python3
import asyncio
import os

import pytest

from rush import Provider
from rush.graphql_client.benchmarks import BenchmarksBenchmarksEdgesNode
from rush.graphql_client.run_benchmark import RunBenchmarkRunBenchmark
from rush.provider import build_provider_with_functions
from rush.graphql_client.create_project import CreateProjectCreateProject

@pytest.fixture
async def provider():
    if "RUSH_TOKEN" in os.environ:
        print("initing")
        return await build_provider_with_functions()
    else:
        print(os.environ)
        print("Skipping test_provider_init")
        return Provider("")

@pytest.fixture
async def project(provider: Provider):
    p = await provider.create_project("test")
    yield p
    await provider.client.delete_project(p.id)

@pytest.fixture
async def benchmark(provider: Provider) -> BenchmarksBenchmarksEdgesNode:
    return (await anext(await provider.benchmarks())).edges[0].node

@pytest.fixture(scope="function")
async def benchmark_submission(provider: Provider, project: CreateProjectCreateProject, benchmark: BenchmarksBenchmarksEdgesNode) -> RunBenchmarkRunBenchmark:
    provider.set_project(project.id)
    return await provider.run_benchmark(benchmark.id, "\\i -> (outputs i)")

@pytest.mark.asyncio
async def test_provider_init(provider: Provider):
    assert provider

@pytest.mark.asyncio
async def test_projects(provider: Provider):
    projects_pages = await provider.projects()
    assert projects_pages is not None
    p = []
    async for page in projects_pages:
        for project in page.edges:
            p.append(project)
    assert len(p) > 0

@pytest.mark.asyncio
async def test_create_project(project: CreateProjectCreateProject):
    assert project is not None
    assert project.name == "test"

@pytest.mark.asyncio
async def test_runs(provider:Provider, project: CreateProjectCreateProject):
    provider.set_project(project.id)
    runs = await provider.runs()
    rs = []
    async for run_page in runs:
        for run in run_page.edges:
            rs.append(run)
    # runs should be zero for new project
    assert len(rs) == 0

@pytest.mark.asyncio
async def test_submit_benchmark(benchmark_submission: RunBenchmarkRunBenchmark):
    assert benchmark_submission is not None

@pytest.mark.asyncio
async def test_runs_after_benchmark_submission(provider: Provider, project: CreateProjectCreateProject, benchmark_submission: RunBenchmarkRunBenchmark):
    provider.set_project(project.id)
    runs = await provider.runs()
    rs = []
    async for run_page in runs:
        for run in run_page.edges:
            rs.append(run)
    assert len(rs) == 1
