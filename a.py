import torch
import torch_test_cpp_extension.msnpu as msnpu_extension

class A(torch.Tensor):
    pass

def foo():
    a = A()
    print(type(a))
    return a

a = torch.empty(5, 5, device='msnpu')
print(type(torch.sigmoid(a)))
