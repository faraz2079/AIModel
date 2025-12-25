import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from loguru import logger


class GPUScheduler:
    """
    Single-GPU cooperative scheduler.

    Guarantees:
    - Only ONE GPU task runs at a time
    - FIFO fairness
    - Explicit queue + execution timing
    - Blocking GPU work runs in a single worker thread
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def run(self, fn):
        """
        fn: synchronous callable that performs GPU work
        """
        logger.info("[SCHEDULER] Request entered scheduler queue")

        async with self._lock:
            logger.info("[SCHEDULER] GPU lock acquired")
            start = time.time()

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self._executor, fn)

            elapsed = time.time() - start
            logger.info(f"[SCHEDULER] GPU task finished in {elapsed:.2f}s")

            return result
