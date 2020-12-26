import numpy as np
from Variable import Variable, numerical_diff, var
from Function import Function, goldstein, rosenbrock
from Calculations import square, exp, add, mul, my_sin
from Context import using_config, no_grad
from utils import plot_dot_graph, _dot_func, _dot_var
import matplotlib.pyplot as plt


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12*x ** 2 - 4

x = var(0)
y = var(2)
iters = 2000
lr = 0.001

for i in range(iters):
    print(x,y)

    z = rosenbrock(x,y)

    x.cleargrad()
    y.cleargrad()
    z.backward()
    x.data -= lr  if x.grad > 0 else -lr
    y.data -= lr  if y.grad > 0 else -lr