"""Lightweight typed handle to a submitted Rush job."""

from __future__ import annotations

import typing
from typing import Any, Generic, TypeVar

from .client import RunID, collect_run

R = TypeVar("R")


class RushRun(Generic[R]):
    """Handle to a submitted Rush job.

    The type parameter *R* is the ``ResultRef`` type returned by
    :meth:`collect`.  Call :meth:`fetch` or :meth:`save` as shorthand for
    ``collect().fetch(...)`` / ``collect().save(...)``.
    """

    def __init__(self, id: RunID, result_type: type[R]) -> None:
        self._id = id
        self._result_type = result_type
        self._collected: R | None = None

    @property
    def id(self) -> RunID:
        return self._id

    def collect(self, max_wait_time: int = 3600) -> R:
        """Wait for the run to complete and return a lightweight result ref.

        The ref is cached: subsequent calls return the same object without
        re-polling the API.
        """
        if self._collected is None:
            raw = collect_run(self._id, max_wait_time=max_wait_time)
            self._collected = typing.cast(Any, self._result_type).from_raw_output(raw)  # type: ignore[attr-defined]
        return self._collected

    def fetch(self, **kwargs: Any) -> Any:
        """Shorthand for ``collect().fetch(**kwargs)``."""
        return typing.cast(Any, self.collect()).fetch(**kwargs)  # type: ignore[union-attr]

    def save(self, **kwargs: Any) -> Any:
        """Shorthand for ``collect().save(**kwargs)``."""
        return typing.cast(Any, self.collect()).save(**kwargs)  # type: ignore[union-attr]

    def __repr__(self) -> str:
        return f"RushRun(id={self._id!r})"
