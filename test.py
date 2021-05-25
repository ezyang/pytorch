import torch

# TODO: TensorBase should work
class A(torch.Tensor):
    x: torch.Tensor

    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        print(f"A new: {cls}, {type(x)}, {x}, {args}, {kwargs}")
        assert not isinstance(x, A)
        assert not x._nontrivial_python
        assert not x.is_meta
        # r = super().__new__(cls, x, *args, **kwargs)
        r = torch.Tensor._make_subclass(cls, x.to('meta'), x.requires_grad)
        assert not isinstance(x, A)
        assert not x._nontrivial_python
        with torch._C.DisableTorchFunction():
            r.x = x
        return r

    def __repr__(self):
        with torch._C.DisableTorchFunction():
            return f"A({self.x})"

    def __str__(self):
        with torch._C.DisableTorchFunction():
            return f"A({self.x})"

    def __format__(self, format_spec):
        with torch._C.DisableTorchFunction():
            return f"A({self.x}, requires_grad={self.requires_grad})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        with torch._C.DisableTorchFunction():
            print(func, args, kwargs)
            if isinstance(func, str):
                unwrapped_args = []
                for a in args:
                    if isinstance(a, A):
                        unwrapped_args.append(a.x)
                    elif isinstance(a, torch.device):
                        if str(a) == "meta":
                            unwrapped_args.append(torch.device("cpu"))
                        else:
                            unwrapped_args.append(a)
                    else:
                        if isinstance(a, torch.Tensor):
                            assert not a.is_meta
                        unwrapped_args.append(a)
                assert not any([a._nontrivial_python or a.is_meta for a in unwrapped_args if isinstance(a, torch.Tensor)])
                new_x = getattr(torch.ops.aten, func.replace("aten::", ""))(*unwrapped_args, **(kwargs if kwargs else {}))
                if isinstance(new_x, torch.Tensor):
                    print("ret tensor")
                    assert not new_x._nontrivial_python
                    assert not new_x.is_meta
                    return [A(new_x)]
                elif isinstance(new_x, list) or isinstance(new_x, tuple):
                    print("ret list")
                    rets = []
                    for sub_new_x in new_x:
                        if isinstance(sub_new_x, torch.Tensor):
                            rets.append(A(sub_new_x))
                    return rets
                else:
                    print("ret other")
                    return [new_x]
            else:
                print("direct entry")
                return func(*args, **(kwargs if kwargs else {}))

def hook(s):
    print(f"s = {s}")
    return s

with torch.autograd.detect_anomaly():
    x = A(torch.tensor([[3.0]], requires_grad=True))
    x.register_hook(hook)
    print(f"x = {x} ({type(x)})")
    # print(f"x.data = {x.data}")
    y = torch.add(x, 2)
    print(f"y = {y} ({type(y)})")
    print("== AD ==")
    # print(torch.autograd.grad((y,), (x,), (A(torch.tensor([[1.0]]),))))
    torch.autograd.backward((y,), (A(torch.tensor([[1.0]]),)))
"""
print("== PRINT ==")
print(x.grad)

print("== CONTROL ==")
x = torch.tensor([[3.0]], requires_grad=True)
y = torch.mul(x, x)
y = torch.log_softmax(y, 0)
y = torch.log_softmax(y, 0)
torch.autograd.backward((y,), (torch.tensor([[1.0]]),))
print(x.grad)
"""
