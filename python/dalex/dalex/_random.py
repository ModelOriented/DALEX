"""Keep explicitly seeded explanation streams local without changing MT19937 draws."""
import numpy as np
from contextlib import contextmanager
import threading

_external_rng_lock = threading.RLock()


def get_random_state(seed):
    # Preserve np.random.seed(...) workflows when callers omit random_state.
    # Explicit seeds must never reset another thread's generator.
    return np.random if seed is None else np.random.RandomState(seed)


@contextmanager
def numpy_state_scope(state):
    """Isolate seeded third-party APIs that only expose NumPy's global RNG.

    DALEX wrapper calls cooperate through this lock. Arbitrary external NumPy
    consumers and unseeded concurrent calls are outside this guarantee.
    """
    with _external_rng_lock:
        if state is None:
            yield
            return
        previous = np.random.get_state()
        try:
            np.random.set_state(state)
            yield
        finally:
            np.random.set_state(previous)
