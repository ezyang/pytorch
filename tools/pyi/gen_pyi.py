from __future__ import print_function
import multiprocessing
import sys
import os
import inspect
import collections
import yaml
import types
import re

needed_modules = set()

FACTORY_PARAMS = "dtype: Optional[dtype]=None, device: Union[device, str, None]=None, requires_grad: bool=False"

# this could be more precise w.r.t list contents etc.
INDICES = "indices: Union[None, int, Ellipsis, slice, Tensor, List, Tuple]"


def type_to_python(typename):
    """type_to_python(typename : str) -> str

Transforms a Declarations.yaml typename into a Python type specification
as used for type hints.
"""
    typename = typename.replace(' ', '')  # some spaces in Generator *
    origtypename = typename
    typename = {
        'Device': 'Union[device, str, None]',
        'Generator*': 'Generator',
        'IntegerTensor': 'Tensor',
        'Scalar': 'Union[float, int]',
        'ScalarType': 'dtype',
        'Storage': 'Storage',
        'BoolTensor': 'Tensor',
        'IndexTensor': 'Tensor',
        'Tensor': 'Tensor',
        'IntList': 'Union[Tuple[int, ...], List[int, ...], Size]',
        'TensorList': 'Union[Tuple[Tensor, ...],List[Tensor, ...]]',
        'bool': 'bool',
        'double': 'float',
        'int64_t': 'int',
        'accreal': 'Union[float, int]',
        'real': 'Union[float, int]',
        'void*': 'int',    # data_ptr
    }[typename]
    return typename


def arg_to_type_hint(arg):
    """arg_to_type_hin(arg) -> str

This takes one argument in a Declarations an returns a string
representing this argument in a type hint signature.
"""
    name = arg['name']
    if name == 'from':  # keyword...
        name += '_'
    typename = type_to_python(arg['dynamic_type'])
    if arg.get('is_nullable'):
        typename = 'Optional[' + typename + ']'
    if 'default' in arg:
        default = arg['default']
        if default == 'nullptr':
            default = None
        default = '={}'.format(default)
    else:
        default = ''
    return name + ': ' + typename + default


def sig_for_ops(opname):
    binary_ops = {'add', 'sub', 'mul', 'div', 'pow', 'lshift', 'rshift', 'mod', 'truediv',
                  'matmul',
                  'radd', 'rmul',                      # reverse arithmetic
                  'and', 'or', 'xor',                  # logic
                  'eq', 'ne', 'ge', 'gt', 'lt', 'le',  # comparisons
                  'iadd', 'iand', 'idiv', 'ilshift', 'imul',
                  'ior', 'irshift', 'isub', 'itruediv', 'ixor',  # inplace ops
                  }
    unary_ops = {'neg', 'abs', 'invert'}
    skip = {'getitem', 'setitem'}
    to_py_type_ops = {'bool', 'float', 'long', 'index', 'int', 'nonzero'}
    # __delitem__,  __new__, __nonzero__, __setitem__
    assert opname.endswith('__') and opname.startswith('__'), "Unexpected op {}".format(opname)
    name = opname[2:-2]
    if name in binary_ops:
        return ['def {}(self, other: Any) -> Tensor: ...'.format(opname)]
    elif name in unary_ops:
        return ['def {}(self) -> Tensor: ...'.format(opname)]
    elif name in skip:
        return []  # expected to be done manually
    elif name in to_py_type_ops:
        if name in {'bool', 'float'}:
            tname = name
        elif name == 'nonzero':
            tname = 'bool'
        else:
            tname = 'int'
        return ['def {}(self) -> {}: ...'.format(opname, tname)]
    else:
        print("### unknown op", opname)
        return []


def generate_type_hints(fname, decls, is_tensor=False):
    """generate_type_hints(fname, decls, is_tensor=False)

Generates type hints for the declarations pertaining to the function
:attr:`fname`. attr:`decls` are the declarations from the parsed
Declarations.yaml.
The :attr:`is_tensor` flag indicates whether we are parsing
members of the Tensor class (true) or functions in the
`torch` namespace (default, false).

This function currently encodes quite a bit about the semantics of
the translation C++ -> Python.
"""
    type_hints = []
    dnames = ([d['name'] for d in decls])
    has_out = fname + '_out' in dnames
    if has_out:
        decls = [d for d in decls if d['name'] != fname + '_out']
    for decl in decls:
        skip = 'Type' in [a['dynamic_type'] for a in decl['arguments']]
        if skip:
            # there currently is one variant of tensor() showing up with a Type argument,
            # Python only uses the TensorOptions one. This check could be taken out once
            # the tensor function is removed
            continue
        render_kw_only_separator = True  # whether we add a '*' if we see a keyword only argument
        python_args = []

        has_tensor_options = 'TensorOptions' in [a['dynamic_type'] for a in decl['arguments']]
        for a in decl['arguments']:
            if a['dynamic_type'] != 'TensorOptions':
                if a.get('kwarg_only', False) and render_kw_only_separator:
                    python_args.append('*')
                    render_kw_only_separator = False
                python_args.append(arg_to_type_hint(a))
        if is_tensor:
            if 'self: Tensor' in python_args:
                python_args.remove('self: Tensor')
                python_args = ['self'] + python_args
            else:
                raise Exception("method without self is unexpected")
        if has_out:
            if render_kw_only_separator:
                python_args.append('*')
                render_kw_only_separator = False
            python_args.append('out: Optional[Tensor]=None')
        if has_tensor_options:
            if render_kw_only_separator:
                python_args.append('*')
                render_kw_only_separator = False
            python_args += ["dtype: dtype=None",
                            "layout: layout=torch.strided",
                            "device: Optional[device]=None",
                            "requires_grad:bool=False"]
        python_args_s = ', '.join(python_args)
        python_returns = [type_to_python(r['dynamic_type']) for r in decl['returns']]
        if len(python_returns) > 1:
            python_returns_s = 'Tuple[' + ', '.join(python_returns) + ']'
        else:
            python_returns_s = python_returns[0]
        type_hint = "def {}({}) -> {}: ...".format(fname, python_args_s, python_returns_s)
        numargs = len(decl['arguments'])
        vararg_pos = int(is_tensor)
        have_vararg_version = (numargs > vararg_pos and
                               decl['arguments'][vararg_pos]['dynamic_type'] in {'IntList', 'TensorList'} and
                               (numargs == vararg_pos + 1 or python_args[vararg_pos + 1] == '*') and
                               (not is_tensor or decl['arguments'][0]['name'] == 'self'))

        type_hints.append(type_hint)
        if have_vararg_version:
            # Two things come into play here: PyTorch has the "magic" that if the first and only positional argument
            # is an IntList or TensorList, it will be used as a vararg variant.
            # The following outputs the vararg variant, the "pass a list variant" is output above.
            # The other thing is that in Python, the varargs are annotated with the element type, not the list type.
            typelist = decl['arguments'][vararg_pos]['dynamic_type']
            if typelist == 'IntList':
                vararg_type = 'int'
            else:
                vararg_type = 'Tensor'
            # replace first argument and eliminate '*' if present
            python_args = ((['self'] if is_tensor else []) + ['*' + decl['arguments'][vararg_pos]['name'] +
                                                              ': ' + vararg_type] + python_args[vararg_pos + 2:])
            python_args_s = ', '.join(python_args)
            type_hint = "def {}({}) -> {}: ...".format(fname, python_args_s, python_returns_s)
            type_hints.append(type_hint)
    return type_hints


def parameters_from_signature(sig):
    """parameters_from_signature(sig) -> str

Takes an inspect.Signature object :attr:`sig` and returns a string
representing the signature (all parameters aka formal arguments)
as we want them to show in the type hint.

For this we iterate through the signature."""
    # is adapted from standard library inspect.Signatur's __str__ function,
    # so it matchesPython's own way of thinking very closely
    result = []
    render_pos_only_separator = False
    render_kw_only_separator = True
    for param in sig.parameters.values():
        kind = param.kind
        formatted = param._name

        # add the type to the signature
        if param._annotation is not inspect._empty:
            formatted = '{}:{}'.format(formatted, inspect.formatannotation(param._annotation))

        if param._default is not inspect._empty:
            # add the default value
            if type(param._default).__name__ == 'module':  # better way?
                # This is for the pickle module passed to the save/load function.
                default_repr = param._default.__name__
                needed_modules.add(default_repr)
            else:
                default_repr = repr(param._default)
            formatted = '{}={}'.format(formatted, default_repr)

        if kind == inspect._VAR_POSITIONAL:
            formatted = '*' + formatted
        elif kind == inspect._VAR_KEYWORD:
            formatted = '**' + formatted

        if kind == inspect._POSITIONAL_ONLY:
            render_pos_only_separator = True
        elif render_pos_only_separator:
            result.append('/')
            render_pos_only_separator = False

        if kind == inspect._VAR_POSITIONAL:
            render_kw_only_separator = False
        elif kind == inspect._KEYWORD_ONLY and render_kw_only_separator:
            result.append('*')
            render_kw_only_separator = False

        formatted = formatted.replace('torch.Tensor', 'Tensor')
        result.append(formatted)

    if render_pos_only_separator:
        result.append('/')
    return result


def type_hint_from_python_fn(fname, fn):
    """type_hint_from_python_fn(fname, fn) -> str

given a function name fname and the function/method object fn,
this function produces the type hint as a string, including
return annotation.
It uses the Python 3 inspect module.
"""
    sig = inspect.signature(fn)
    python_parameters = parameters_from_signature(sig)

    return_annotation = None
    if sig.return_annotation is not inspect._empty:
        return_annotation = inspect.formatannotation(sig.return_annotation)
        return_annotation = return_annotation.replace('torch.Tensor', 'Tensor')

    python_parameters = ', '.join(python_parameters)
    if return_annotation:
        return ["def {}({}) -> {}: ...".format(fname, python_parameters, return_annotation)]
    else:
        return ["def {}({}): ...".format(fname, python_parameters, return_annotation)]


def do_gen_pyi(build_lib_path):
    """do_gen_pyi(build_lib_path)

This function generates a pyi file for torch. It does so by removing things from the
import path and adding the freshly build PyTorch.
As such it is inteded to be used from a subprocess.
"""
    while '' in sys.path:
        # we don't want to have the source directory (with a torch but without torch._C) in our path
        sys.path.remove('')
    sys.path.insert(0, build_lib_path)
    import torch
    assert torch.__file__.startswith(build_lib_path)

    fns = collections.defaultdict(list)
    yaml_loader = getattr(yaml, 'CLoader', yaml.loader)

    for d in yaml.load(open('torch/share/ATen/Declarations.yaml'),
                       Loader=yaml_loader):

        name = d['name']
        if name.endswith('_out'):
            name = name[:-4]
        if not name.startswith('_'):
            fns[name].append(d)

    type_hints = collections.defaultdict(list)
    type_hints.update({
        'tensor': ["def tensor(data: Any, {}) -> Tensor: ...".format(FACTORY_PARAMS)],
        'set_flush_denormal': ['def set_flush_denormal(mode: bool) -> bool: ...'],
        'get_default_dtype': ['def get_default_dtype() -> dtype: ...'],
        'from_numpy': ['def from_numpy(ndarray) -> Tensor: ...'],
        'clamp': ["def clamp(self, min: float=-math.inf, max: float=math.inf, out: Optional[Tensor]=None) "
                  "-> Tensor: ..."],
        'as_tensor': ["def as_tensor(data: Any, dtype: dtype=None, device: Optional[device]=None) -> Tensor: ..."],
    })

    for fname in dir(torch):
        fn = getattr(torch, fname)
        docstr = inspect.getdoc(fn)
        if docstr:
            if isinstance(fn, types.BuiltinFunctionType):
                if fname in fns:
                    type_hints[fname] += generate_type_hints(fname, fns[fname])
                elif fname not in type_hints:
                    # if we have annotated them manually, we can skip them here without worrying too much
                    print("### skipping", fname)  # todos
            elif isinstance(fn, types.FunctionType):
                type_hints[fname] += type_hint_from_python_fn(fname, fn)

    type_hints_list = []
    for fname, hints in sorted(type_hints.items()):
        if len(hints) > 1:
            hints = ['@overload\n' + h for h in hints]
        type_hints_list += hints
    type_hints_s = '\n\n'.join(type_hints_list) + '\n'

    tensor_type_hints = collections.defaultdict(list)
    tensor_type_hints.update({
        'size': ['def size(self) -> Size: ...'],
        'stride': ['def stride(self) -> Tuple[int]: ...'],
        'new_empty': ['def new_empty(self, size: {}, {}) -> Tensor: ...'.
                      format(type_to_python('IntList'), FACTORY_PARAMS)],
        'new_ones': ['def new_ones(self, size: {}, {}) -> Tensor: ...'.
                     format(type_to_python('IntList'), FACTORY_PARAMS)],
        'new_zeros': ['def new_zeros(self, size: {}, {}) -> Tensor: ...'.
                      format(type_to_python('IntList'), FACTORY_PARAMS)],
        'new_full': ['def new_full(self, size: {}, value: {}, {}) -> Tensor: ...'.
                     format(type_to_python('IntList'), type_to_python('Scalar'), FACTORY_PARAMS)],
        'new_tensor': ["def new_tensor(self, data: Any, {}) -> Tensor: ...".format(FACTORY_PARAMS)],
        'clamp': ["def clamp(self, min: float=-math.inf, max: float =math.inf) -> Tensor: ..."],
        'clamp_': ["def clamp(self, min: float=-math.inf, max: float =math.inf) -> Tensor: ..."],
        '__getitem__': ["def __getitem__(self, {}) -> Tensor: ...".format(INDICES)],
        '__setitem__': ["def __setitem__(self, {}, val: Union[Tensor, float, int]) -> None: ...".format(INDICES)],
        'item': ['def item(self) -> Union[float, int]: ...'],
        'tolist': ['def tolist(self) -> List: ...'],
        'requires_grad_': ['def requires_grad_(mode: bool) -> Tensor: ...'],
        'element_size': ['def element_size() -> int: ...'],
        'dim': ['def dim() -> int: ...'],
        'ndimension': ['def ndimension() -> int: ...'],
        'nelement': ['def nelement() -> int: ...'],
        'cuda': ['def cuda(device: Optional[device]=None, non_blocking: bool=False) -> Tensor: ...'],
        'numpy': ['def numpy() -> Any: ...'],
    })
    simple_conversions = ['byte', 'char', 'cpu', 'double', 'float', 'half', 'int', 'long', 'short']
    for fname in simple_conversions:
        tensor_type_hints[fname].append('def {}() -> Tensor: ...'.format(fname))

    for fname in dir(torch.Tensor):
        fn = getattr(torch.Tensor, fname)
        docstr = inspect.getdoc(fn)
        if (docstr and not fname.startswith('_')) or fname.startswith('__'):
            if getattr(fn, '__qualname__', '').startswith('_TensorBase.'):  # better check?
                if fname in fns:
                    tensor_type_hints[fname] += generate_type_hints(fname, fns[fname], is_tensor=True)
                elif fname.startswith('__') and fname.endswith('__'):
                    sig = sig_for_ops(fname)
                    if sig:
                        tensor_type_hints[fname] += sig
                elif fname not in tensor_type_hints:  # it's presumably OK if we already have a type hint
                    print("### skipping Tensor.", fname)
            elif isinstance(fn, types.FunctionType):  # python defined
                tensor_type_hints[fname] += type_hint_from_python_fn(fname, fn)

    type_hints_list = []
    for fname, hints in sorted(tensor_type_hints.items()):
        if len(hints) > 1:
            type_hints_list += ['    @overload\n    ' + re.sub(r"\bTensor\b", "'Tensor'", h) for h in hints]
        else:
            type_hints_list.append('    ' + re.sub(r"\bTensor\b", "'Tensor'", hints[0]))  # it is only one
    tensor_type_hints_s = """class Tensor:
    dtype: dtype = ...
    shape: Size = ...
    requires_grad: bool = ...
    grad: Optional['Tensor'] = ...

""" + '\n\n'.join(type_hints_list) + '\n\n'

    header = """
from typing import List, Tuple, Optional, Union, Any, ContextManager, overload

"""
    header += '\n'.join(["import " + m for m in needed_modules]) + '\n\n'

    header += """
class dtype: ...

class layout: ...

class device:
   def __init__(self, device: Union[device, str, None]=None) -> None: ...

class Generator: ...

class Size(tuple): ...

class enable_grad():
    def __enter__(self) -> None: ...
    def __exit__(self, *args) -> None: ...
    def __call__(self, func : Callable) -> Callable: ...

class no_grad():
    def __enter__(self) -> None: ...
    def __exit__(self, *args) -> None: ...
    def __call__(self, func : Callable) -> Callable: ...

class set_grad_enabled():
    def __init__(self, mode: bool) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args) -> None: ...


"""
    header += '\n'.join(['{}: dtype = ...'.format(n)
                         for n in dir(torch) if isinstance(getattr(torch, n), torch.dtype)])

    footer_classes = []
    for c in ('DoubleStorage', 'FloatStorage', 'LongStorage', 'IntStorage',
              'ShortStorage', 'CharStorage', 'ByteStorage'):
        footer_classes.append('class {}(Storage): ...'.format(c))
    for c in ('DoubleTensor', 'FloatTensor', 'LongTensor', 'IntTensor',
              'ShortTensor', 'CharTensor', 'ByteTensor'):
        footer_classes.append('class {}(Tensor): ...'.format(c))

    footer = '\n\n' + '\n\n'.join(footer_classes)

    with open(os.path.join(build_lib_path, 'torch', '__init__.pyi'), 'w') as f:
        print(header, file=f)
        print(tensor_type_hints_s, file=f)
        print(type_hints_s, file=f)
        print(footer, file=f)


def gen_pyi(build_lib_path):
    # we import torch, better do that in a subprocess
    p = multiprocessing.Process(target=do_gen_pyi, args=(build_lib_path,))
    p.start()
    p.join()
