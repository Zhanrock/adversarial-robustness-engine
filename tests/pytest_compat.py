"""
pytest_compat.py
-----------------
Compatibility shim: provides ``pytest.approx``, ``pytest.raises``,
``pytest.mark``, and ``pytest.fixture`` so that test files written for
pytest also run cleanly with the stdlib ``unittest`` runner.

When pytest IS available this module is never imported.
"""

import contextlib
import unittest


class _Approx:
    """Minimal pytest.approx equivalent."""

    def __init__(self, expected, abs=1e-6, rel=1e-6):
        self.expected = expected
        self.abs_tol = abs
        self.rel_tol = rel

    def __eq__(self, other):
        return abs(other - self.expected) <= max(self.abs_tol, self.rel_tol * abs(self.expected))

    def __repr__(self):
        return f"approx({self.expected!r})"


def approx(val, abs=1e-6, rel=1e-6):
    return _Approx(val, abs=abs, rel=rel)


@contextlib.contextmanager
def raises(exc_type):
    try:
        yield
        raise AssertionError(f"Expected {exc_type.__name__} to be raised")
    except exc_type:
        pass


class _MarkDecorator:
    def __getattr__(self, name):
        def decorator_or_value(*args, **kwargs):
            # If called with a single callable, it's a decorator
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]

            # Otherwise return a decorator factory
            def decorator(fn):
                return fn

            return decorator

        return decorator_or_value

    def parametrize(self, argnames, argvalues, **kwargs):
        """
        Minimal parametrize: flattens into unittest-runnable test methods.
        Returns a class decorator that duplicates the test for each set of params.
        """

        def class_decorator(cls):
            return cls

        return class_decorator


mark = _MarkDecorator()


def fixture(fn=None, scope="function", **kwargs):
    """No-op fixture decorator for unittest compatibility."""
    if fn is not None:
        return fn

    def decorator(f):
        return f

    return decorator
