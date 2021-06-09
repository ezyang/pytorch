import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from typing import Iterator, List, Dict
import logging
import contextlib

# TODO: TensorBase should work
class LoggingTensor(torch.Tensor):
    elem: torch.Tensor

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        r = torch.Tensor._make_subclass(cls, elem.to('meta'), elem.requires_grad)
        r.elem = elem
        return r

    def __repr__(self):
        return f"LoggingTensor({self.elem})"

    def __str__(self):
        return f"LoggingTensor({self.elem})"

    def __format__(self, format_spec):
        return f"LoggingTensor({self.elem})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        with torch._C.DisableTorchFunction():
            if hasattr(func, '__module__') and func.__module__.startswith("torch._ops."):
                assert not kwargs
                unwrapped_args = []
                for a in args:
                    if isinstance(a, LoggingTensor):
                        unwrapped_args.append(a.elem)
                    elif isinstance(a, torch.device):
                        if str(a) == "meta":
                            unwrapped_args.append(torch.device("cpu"))
                        else:
                            unwrapped_args.append(a)
                    else:
                        unwrapped_args.append(a)
                r = func(*unwrapped_args)
                if isinstance(r, torch.Tensor):
                    rs = [LoggingTensor(r)]
                elif isinstance(r, list) or isinstance(r, tuple):
                    rs = [LoggingTensor(e) if isinstance(e, torch.Tensor) else e for e in r]
                else:
                    rs = [r]
                logging.getLogger("LoggingTensor").info(f"{func.__module__}.{func.__name__}", args, rs)
                return rs
            else:
                return func(*args, **(kwargs if kwargs else {}))

# https://stackoverflow.com/questions/36408496/python-logging-handler-to-append-to-list
class LoggingTensorHandler(logging.Handler):
    log_list: List[str]
    next_shortid: int
    id2shortid: Dict[int, int]

    def __init__(self, log_list: List[str]) -> None:
        logging.Handler.__init__(self)
        self.log_list = log_list
        self.next_shortid = 0
        self.id2shortid = {}

    # WARNING: not thread safe
    def _shortid(self, o: object) -> int:
        if id(o) not in self.id2shortid:
            self.id2shortid[id(o)] = self.next_shortid
            self.next_shortid += 1
        return self.id2shortid.get(id(o))

    def _fmt(self, a: object) -> str:
        return f'${self._shortid(a)}' if isinstance(a, LoggingTensor) else repr(a)

    def emit(self, record):
        fmt_args = "(" + ", ".join(self._fmt(a) for a in record.args[0]) + ")"
        fmt_rets = ", ".join(self._fmt(a) for a in record.args[1])
        self.log_list.append(f'{fmt_rets} = {record.msg}{fmt_args}')

def log_input(name: str, var: object):
    logging.getLogger("LoggingTensor").info("input", (name,), (var,))

@contextlib.contextmanager
def capture_logs() -> Iterator[List[str]]:
    logger = logging.getLogger("LoggingTensor")
    log_list = []
    handler = LoggingTensorHandler(log_list)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield log_list
    finally:
        logger.removeHandler(handler)

class TestPythonDispatch(TestCase):
    def test_basic(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.tensor([3.0], requires_grad=True))
            log_input("x", x)
            y = x * x
            grad_y = LoggingTensor(torch.tensor([1.0]))
            log_input("grad_y", grad_y)
            g, = torch.autograd.grad((y,), (x,), (grad_y,))

        self.assertEqual(g.elem, torch.tensor([6.0]))
        # $3 and $5 comes from nowhere because detach for saved variable not tracked atm
        self.assertExpectedInline('\n'.join(logs), '''\
$0 = input('x')
$1 = torch._ops.aten.mul($0, $0)
$2 = input('grad_y')
$4 = torch._ops.aten.mul($2, $3)
$6 = torch._ops.aten.mul($2, $5)
$7 = torch._ops.aten.add($6, $4, 1)''')

if __name__ == '__main__':
    run_tests()
