import asyncio
from collections import deque
from typing import Callable, Any

class GPUScheduler:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._queue = deque()
        self._queue_lock = asyncio.Lock()

    async def run(self, fn: Callable[[], Any]) -> Any:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        async with self._queue_lock:
            self._queue.append(fut)

        # FIFO fairness
        while True:
            async with self._queue_lock:
                if self._queue and self._queue[0] is fut:
                    break
            await asyncio.sleep(0.001)

        async with self._lock:
            try:
                # run blocking GPU work safely
                result = await loop.run_in_executor(None, fn)
                return result
            finally:
                async with self._queue_lock:
                    self._queue.popleft()
