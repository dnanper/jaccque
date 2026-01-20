"""Async utilities for running async code in sync contexts.

Handles the complexity of running async memory operations
from LangGraph's synchronous node callbacks.

The key issue is that LangGraph nodes run synchronously, but
asyncpg (used by MemoryEngine) requires a persistent event loop.
We solve this by maintaining a dedicated thread with its own event loop.
"""

import asyncio
import atexit
import logging
import threading
from concurrent.futures import Future
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncRunner:
    """Runs async functions from sync code using a dedicated thread+loop.
    
    This avoids the "Event loop is closed" error that occurs when
    using asyncio.run() multiple times with asyncpg connections.
    
    Usage:
        runner = AsyncRunner()
        result = runner.run(some_async_function())
        runner.close()
    """
    
    _instance: "AsyncRunner | None" = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started = False
    
    @classmethod
    def get_instance(cls) -> "AsyncRunner":
        """Get or create the singleton AsyncRunner instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance.start()
                    # Register cleanup on exit
                    atexit.register(cls._instance.close)
        return cls._instance
    
    def start(self) -> None:
        """Start the background event loop thread."""
        if self._started:
            return
        
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="AsyncRunner-EventLoop"
        )
        self._thread.start()
        self._started = True
        logger.debug("AsyncRunner started")
    
    def _run_loop(self) -> None:
        """Run the event loop in background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
    
    def run(self, coro: Coroutine[Any, Any, T], timeout: float = 30.0) -> T:
        """Run an async coroutine from sync code.
        
        Args:
            coro: Async coroutine to run
            timeout: Timeout in seconds
            
        Returns:
            Result of the coroutine
            
        Raises:
            TimeoutError: If coroutine doesn't complete in time
            Exception: Any exception raised by the coroutine
        """
        if not self._started or self._loop is None:
            self.start()
        
        # Submit to background loop
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            future.cancel()
            raise
    
    def close(self) -> None:
        """Stop the background event loop and thread."""
        if not self._started:
            return
        
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        self._started = False
        logger.debug("AsyncRunner closed")


def run_async(coro: Coroutine[Any, Any, T], timeout: float = 30.0) -> T:
    """Run an async coroutine from sync code.
    
    This is the main entry point for calling async memory operations
    from LangGraph's synchronous nodes.
    
    Uses a singleton AsyncRunner with a persistent event loop,
    avoiding the "Event loop is closed" errors.
    
    Args:
        coro: Async coroutine to run
        timeout: Timeout in seconds
        
    Returns:
        Result of the coroutine
        
    Example:
        from agent.memory.async_utils import run_async
        
        # In a sync node
        result = run_async(memory_client.recall(query))
    """
    return AsyncRunner.get_instance().run(coro, timeout)


def close_async_runner() -> None:
    """Close the singleton AsyncRunner.
    
    Call this during cleanup to ensure clean shutdown.
    """
    if AsyncRunner._instance:
        AsyncRunner._instance.close()
        AsyncRunner._instance = None


__all__ = ["run_async", "close_async_runner", "AsyncRunner"]
