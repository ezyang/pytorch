from copyreg import dispatch_table
import copy

# Backport of 3.11 object.__getstate__ implementation
def default_getstate(x):
    # xref: https://github.com/python/cpython/blob/f04c16875b649e2c2b420eb146308d0206c7e527/Objects/typeobject.c#L5883-L6006
    state = getattr(x, '__dict__', None)
    if not state:
        state = None

    slotnames = getattr(x, '__slots__', None)
    # NB: required is False
    if slotnames is not None and len(slotnames) > 0:
        slots = {}
        for name in slotnames:
            if hasattr(x, name):
                slots[name] = getattr(x, name)
            # It is not an error if the attribute is not present

    if slots:
        return (state, slots)
    return state

# This function reimplements what copy.deepcopy does if __deepcopy__
# is not define.  This can be used if you need to implement __deepcopy__
# but sometimes you want to fallback to the default behavior
# (super().__deepcopy__ would not work as it is not defined on the
# superclass.)
def default_deepcopy(x, memo):
    cls = type(x)

    # xref: https://github.com/python/cpython/blob/f04c16875b649e2c2b420eb146308d0206c7e527/Lib/copy.py#L145-L162
    reductor = dispatch_table.get(cls)
    if reductor:
        rv = reductor(x)
    else:
        reductor = getattr(x, "__reduce_ex__", None)
        if reductor is not None:
            rv = reductor(4)
        else:
            reductor = getattr(x, "__reduce__", None)
            if reductor:
                rv = reductor()
            else:
                raise copy.Error(
                    "un(deep)copyable object of type %s" % cls)
    if isinstance(rv, str):
        y = x
    else:
        y = copy._reconstruct(x, memo, *rv)

    # NB: We don't have to handle memoization; __deepcopy__ handler deals
    # with this for us too.
    return y
