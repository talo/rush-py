"""Lightweight typed handle to a submitted Rush job."""

from __future__ import annotations

from typing import Any, Generic, Protocol, TypeVar, cast

from .client import RunID, collect_run


R = TypeVar("R")


class _FromRawOutput(Protocol[R]):
    @classmethod
    def from_raw_output(cls, raw: Any) -> R: ...


class _FetchableSavable(Protocol):
    def fetch(self, **kwargs: Any) -> Any: ...

    def save(self, **kwargs: Any) -> Any: ...


class RushRun(Generic[R]):
    """Handle to a submitted Rush job.

    The type parameter *R* is the ``ResultRef`` type returned by
    :meth:`collect`. Call :meth:`fetch` or :meth:`save` as shorthand for
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
            result_factory = cast(type[_FromRawOutput[R]], self._result_type)
            self._collected = result_factory.from_raw_output(raw)
        return self._collected

    def fetch(self, **kwargs: Any) -> Any:
        """Shorthand for ``collect().fetch(**kwargs)``."""
        result_ref = cast(_FetchableSavable, self.collect())
        return result_ref.fetch(**kwargs)

    def save(self, **kwargs: Any) -> Any:
        """Shorthand for ``collect().save(**kwargs)``."""
        result_ref = cast(_FetchableSavable, self.collect())
        return result_ref.save(**kwargs)

    def __repr__(self) -> str:
        return f"RushRun(id={self._id!r})"
