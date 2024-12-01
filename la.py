import torch
import os.path
import pickle
from torch._inductor.codecache import FxGraphCache
from torch._functorch._aot_autograd.schemas import AOTConfig

"""
key = "fg5kds52hrpjjbroe6zggnzdzckhney24oomlcxghyxozzxzf5fd"

subdir = FxGraphCache._get_tmp_dir_for_key(key)
r = []
for path in sorted(os.listdir(subdir)):
    with open(os.path.join(subdir, path), "rb") as f:
        r.append(pickle.load(f))
print(r)
"""

aot_autograd_key = "azpz2jolskvr5764aawh4yx7kek6j5sljqzr5eicpof6acvg5kbp"

from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache

entry = AOTAutogradCache._lookup(aot_autograd_key, local=True, remote=False)

aot_config = AOTConfig(
    fw_compiler=None,
    bw_compiler=None,
    inference_compiler=None,
    partition_fn=None,
    decompositions=None,
    num_params_buffers=0,
    aot_id=0,
    keep_inference_input_mutations=False,
    dynamic_shapes=False,
    aot_autograd_arg_pos_to_source=None,
    is_export=False,
    no_tangents=False,
    enable_log=False,
)
fx_config = {"cudagraphs": None}

args = [torch.randn(2, device="cuda", requires_grad=True), torch.randn(2, device="cuda", requires_grad=True)]
compiled_fn = entry.wrap_post_compile(args, aot_config, fx_config)

print(compiled_fn)
print(compiled_fn(args))
