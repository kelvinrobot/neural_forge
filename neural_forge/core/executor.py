"""Thread pool executor for blocking model inference."""

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

ResultType = TypeVar("ResultType")


class InferenceExecutor:
    """Runs blocking inference and image processing off the event loop."""

    def __init__(self, max_workers: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nf-inference")
        self._max_workers = max_workers

    @property
    def max_workers(self) -> int:
        """Number of worker threads."""
        return self._max_workers

    async def run(self, function: Callable[..., ResultType], *args: Any, **kwargs: Any) -> ResultType:
        """Execute a blocking callable in the pool and await its result."""
        loop = asyncio.get_running_loop()
        bound_call = functools.partial(function, *args, **kwargs)
        return await loop.run_in_executor(self._pool, bound_call)

    def shutdown(self) -> None:
        """Stop accepting work and release worker threads."""
        self._pool.shutdown(wait=False, cancel_futures=True)
