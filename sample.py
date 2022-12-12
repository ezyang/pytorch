import torch
import torch._dynamo
import torch._dynamo.config
from torch._dynamo.optimizations.training import aot_autograd
from torch._dynamo.utils import fake_mode_from_tensors
import logging
from torch import SymInt
from typing import NamedTuple, List
import torch.nn.functional as F

class TensorMeta(NamedTuple):
    size: List[int]
    stride: List[int]
    storage_offset: int

# Bug reports:
# - AOT_FX_GRAPHS_JOINT=1 no longer works
# - torch.compile doesn't take string backend name
# - Requires torch._dynamo manually imported for torch.compile to work
# - Should aot_autograd by default
# - aot_autograd signature is annoying (accept fw_compiler as positional arg)
# - make_boxed_func warning points to wrong module
# - make_boxed_func annoying for idiomatic return gm.forward

# Need to make this more compact
# - Hash consing to CSE as we go
# - Alternate output format that's not FX (binary maybe?)

def my_compiler(gm, example_inputs):
    fake_mode = fake_mode_from_tensors(example_inputs)
    # TODO: get this from tracing context
    assert fake_mode is not None

    """
    def print_t(t):
        print(
            tuple(s.node.fx_node if isinstance(s, SymInt) else s for s in t.size()),
            tuple(s.node.fx_node if isinstance(s, SymInt) else s for s in t.stride()),
            t.storage_offset().node.fx_node if isinstance(t.storage_offset(), SymInt) else t.storage_offset()
        )
    for n in gm.graph.nodes:
        if n.op == "output":
            for out in n.args[0]:
                t = out.meta['val']
                # TODO: handle symint out too
                assert isinstance(t, torch.Tensor)
                print_t(t)
    """

    shape_tracer = fake_mode.shape_env.fx_tracer
    shape_graph = shape_tracer.graph
    shape_outputs = []
    for n in gm.graph.nodes:
        if n.op == "output":
            for out in n.args[0]:
                t = out.meta['val']
                # TODO: handle symint out too
                assert isinstance(t, torch.Tensor)
                shape_outputs.append(TensorMeta(
                    size=[s.node.fx_node if isinstance(s, SymInt) else s for s in t.size()],
                    stride=[s.node.fx_node if isinstance(s, SymInt) else s for s in t.stride()],
                    storage_offset=t.storage_offset().node.fx_node if isinstance(t.storage_offset(), SymInt) else t.storage_offset()
                ))

    shape_tracer.create_node("output", "", (shape_outputs,), {})
    # Better not to DCE
    # shape_graph.eliminate_dead_code()

    torch.fx.GraphModule({}, shape_graph).print_readable()
    #print(gm)
    #print(example_inputs)
    #for g in fake_mode.shape_env.fx_guards:
    #    print("guard", g)
    return torch._functorch.aot_autograd.make_boxed_func(gm.forward)

# Hmm what about backwards?

aot_my_compiler = aot_autograd(fw_compiler=my_compiler)

@torch._dynamo.optimize(aot_my_compiler, dynamic=True)
def f(a, b):
    return a @ b

f(torch.randn(20, 30, requires_grad=True), torch.randn(30, 40))


@torch._dynamo.optimize(aot_my_compiler, dynamic=True)
def f2(inputs, filters):
    return F.conv1d(inputs, filters)

inputs = torch.randn(33, 16, 30)
filters = torch.randn(20, 16, 5)
f2(inputs, filters)
