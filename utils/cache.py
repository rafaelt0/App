import copy
import functools
import hashlib
import inspect
import logging
import pickle
import threading
import time
from collections import OrderedDict


logger = logging.getLogger(__name__)
_caches = []


def cache_data(ttl=3600, max_entries=128, stale=0):
    """Memoize by arguments for ``ttl`` seconds, one computation per key at a time.

    With ``stale`` > 0, an entry up to ``ttl + stale`` seconds old is returned
    immediately while one background thread recomputes it, so readers wait
    only when no usable entry exists.
    """
    def decorate(function):
        signature = inspect.signature(function)
        entries = OrderedDict()
        pending = {}
        lock = threading.RLock()
        generation = 0

        def key_for(args, kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = {key: value for key, value in bound.arguments.items() if not key.startswith("_")}
            return hashlib.sha256(pickle.dumps(values, protocol=5)).digest()

        def compute(key, event, started_generation, args, kwargs):
            try:
                value = function(*args, **kwargs)
                with lock:
                    if generation == started_generation:
                        entries[key] = (time.monotonic(), copy.deepcopy(value))
                        entries.move_to_end(key)
                        while len(entries) > max_entries:
                            entries.popitem(last=False)
                return value
            finally:
                with lock:
                    pending.pop(key, None)
                    event.set()

        def revalidate(*arguments):
            try:
                compute(*arguments)
            except Exception:
                logger.warning("Background refresh of %s failed; keeping the stale entry", function.__qualname__, exc_info=True)

        @functools.wraps(function)
        def cached(*args, **kwargs):
            key = key_for(args, kwargs)
            while True:
                with lock:
                    entry = entries.get(key)
                    age = None if entry is None else time.monotonic() - entry[0]
                    if age is not None and age < ttl + stale:
                        entries.move_to_end(key)
                        if age >= ttl and key not in pending:
                            event = pending[key] = threading.Event()
                            threading.Thread(target=revalidate, args=(key, event, generation, args, kwargs),
                                             name=f"refresh-{function.__name__}", daemon=True).start()
                        return copy.deepcopy(entry[1])
                    entries.pop(key, None)
                    event = pending.get(key)
                    if event is None:
                        event = pending[key] = threading.Event()
                        started_generation = generation
                        break
                event.wait()
            return compute(key, event, started_generation, args, kwargs)

        def clear(*args, **kwargs):
            nonlocal generation
            with lock:
                generation += 1
                if args or kwargs:
                    entries.pop(key_for(args, kwargs), None)
                else:
                    entries.clear()

        cached.clear = clear
        _caches.append(clear)
        return cached
    return decorate


def clear_all():
    for clear in _caches:
        clear()
