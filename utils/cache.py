import copy
import functools
import hashlib
import inspect
import pickle
import threading
import time
from collections import OrderedDict


_caches = []


def cache_data(ttl=3600, max_entries=128):
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

        @functools.wraps(function)
        def cached(*args, **kwargs):
            key = key_for(args, kwargs)
            while True:
                with lock:
                    now = time.monotonic()
                    entry = entries.get(key)
                    if entry is not None and now - entry[0] < ttl:
                        entries.move_to_end(key)
                        return copy.deepcopy(entry[1])
                    entries.pop(key, None)
                    event = pending.get(key)
                    if event is None:
                        event = threading.Event()
                        pending[key] = event
                        started_generation = generation
                        break
                event.wait()
            try:
                value = function(*args, **kwargs)
                with lock:
                    if generation == started_generation:
                        entries[key] = (time.monotonic(), copy.deepcopy(value))
                        while len(entries) > max_entries:
                            entries.popitem(last=False)
                return value
            finally:
                with lock:
                    pending.pop(key, None)
                    event.set()

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
