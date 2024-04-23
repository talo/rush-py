#!/usr/bin/env python3
import asyncio
import os

from rush import Provider
from rush.provider import build_provider_with_functions


def test_provider_init():
    # only run if RUSH_TOKEN is set
    if "RUSH_TOKEN" in os.environ:
        asyncio.run(build_provider_with_functions())
    Provider("")
