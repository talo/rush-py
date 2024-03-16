#!/usr/bin/env python3

import asyncio
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def start_background_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


LOOP = asyncio.new_event_loop()


def asyncio_run(coro: Coroutine[Any, Any, T]) -> T:
    """
    Runs the coroutine in an event loop running on a background thread,
    and blocks the current thread until it returns a result.
    This plays well with gevent, since it can yield on the Future result call.

    :param coro: A coroutine, typically an async method
    :param timeout: How many seconds we should wait for a result before raising an error
    """
    if LOOP.is_running():
        return asyncio.run_coroutine_threadsafe(coro, LOOP).result()
    else:
        return asyncio.create_task(coro)
