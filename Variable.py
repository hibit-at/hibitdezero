import numpy as np

class Variable:
    def __init__(self, data, name=None):
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation+1

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        # funcs = [self.creator]

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

        if not retain_grad:
            for y in f.outputs:
                pass
                # y().grad = None  # yはweakref

    def cleargrad(self):
        self.grad = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def stype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n','\n' + ' '*9)
        return 'variable(' + p + ')'

    def __mul__(self, other):
        from Calculations import mul
        return mul(self, other)

    def __add__(self, other):
        from Calculations import add
        return add(self, other)

    def __rmul__(self, other):
        from Calculations import mul
        return mul(self, other)

    def __radd__(self, other):
        from Calculations import add
        return add(self, other)

    def __neg__(self):
        from Calculations import neg
        return neg(self)

    def __sub__(self, other):
        from Calculations import sub
        return sub(self, other)

    def __rsub__(self, other):
        from Calculations import rsub
        return rsub(self, other)

    def __pow__(self, other):
        from Calculations import pow
        return pow(self, other)

def var(x):
    return Variable(np.array([x]))


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x