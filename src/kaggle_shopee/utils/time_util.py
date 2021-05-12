import contextlib
import functools
import math
from typing import Callable

from kaggle_shopee.utils.global_util import GlobalUtil


class TimeUtil:
    @staticmethod
    @contextlib.contextmanager
    def timer(name: str):
        t0, m0, p0 = GlobalUtil.get_metric()
        print(f"[{name}] start [{m0:.1f}GB({p0:.1f}%)]")
        yield
        t1, m1, p1 = GlobalUtil.get_metric()
        delta = m1 - m0
        sign = "+" if delta >= 0 else "-"
        delta = math.fabs(delta)
        print(
            f"[{name}] done [{m1:.1f}GB({p1:.1f}%)({sign}{delta:.3f}GB)] {t1 - t0:.4f} s"
        )

    @staticmethod
    def timer_wrapper(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0, m0, p0 = GlobalUtil.get_metric()
            print(f"[{func.__name__}] start [{m0:.1f}GB({p0:.1f}%)]")
            value = func(*args, **kwargs)
            t1, m1, p1 = GlobalUtil.get_metric()
            delta = m1 - m0
            sign = "+" if delta >= 0 else "-"
            delta = math.fabs(delta)
            print(
                f"[{func.__name__}] done [{m1:.1f}GB({p1:.1f}%)({sign}{delta:.3f}GB)] {t1 - t0:.4f} s"
            )
            return value

        return wrapper
