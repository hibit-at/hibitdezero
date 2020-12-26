from Variable import Variable, as_variable, as_array
import weakref
from Config import Config


class Function:

    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs  # 入力された関数を覚える
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


def goldstein(x, y):
    z = (1 + (x+y+1)**2 * (19-14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (38 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))

    return z


def rosenbrock(x0, x1):
    y = 100*(x1-x0**2)**2 + (x0-1)**2
    return y



