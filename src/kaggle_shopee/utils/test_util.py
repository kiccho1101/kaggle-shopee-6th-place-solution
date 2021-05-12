import functools
from typing import Any, Callable


class TestUtil:
    @staticmethod
    def assert_any(actual: Any, expect: Any):
        if actual != actual or expect != expect:
            assert (
                actual != actual and expect != expect
            ), f"Expect: {expect} Actual: {actual}"
        else:
            assert type(actual) == type(
                expect
            ), f"Expect type: {type(expect)}, Actual type: {type(actual)}"
            assert actual == expect, (
                "Expect: {} Actual: {}, difference: {}".format(
                    expect,
                    actual,
                    set(actual).union(set(expect))
                    - set(actual).intersection(set(expect)),
                )
                if isinstance(actual, list)
                else f"Expect: {expect} Actual: {actual}"
            )

    @staticmethod
    def test_data(func: Callable):
        @functools.wraps(func)
        def wrapper(data, *args, **kwargs):
            train_len_before = len(data.train)
            test_len_before = len(data.test)
            value = func(data, *args, **kwargs)
            train_len_after = len(data.train)
            test_len_after = len(data.test)
            TestUtil.assert_any(train_len_after, train_len_before)
            TestUtil.assert_any(test_len_after, test_len_before)
            return value

        return wrapper
