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
    try:
        # FIXME: this is a hack to work around an annoying stall issue in Jupyter notebooks
        #        for some reason, run_coroutine_threadsafe hangs indefinitely when uploading files
        #        inside the jupyter notebook. nest_asyncio allows us to just use LOOP.run_until_complete
        __IPYTHON__  # noqa: F821
        import nest_asyncio

        nest_asyncio.apply()
    except Exception:
        pass

    if LOOP.is_running():
        return asyncio.run_coroutine_threadsafe(coro, LOOP).result()
    elif asyncio.get_event_loop().is_running():
        r = asyncio.run(coro)
        return r
    else:
        try:
            return LOOP.run_until_complete(coro)
        except RuntimeError:
            start_background_loop(LOOP)
            return asyncio.run_coroutine_threadsafe(coro, LOOP).result()
        except Exception as e:
            print(e)
