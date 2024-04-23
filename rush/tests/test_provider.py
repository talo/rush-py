#!/usr/bin/env python3
import asyncio

from rush import Provider
from rush.provider import build_provider_with_functions


def test_provider_init():
    asyncio.run(build_provider_with_functions())
    Provider("")
