import numpy as np
from Variable import Variable, numerical_diff, var
from Function import Function, goldstein, rosenbrock
from Calculations import square, exp, add, mul, my_sin, sin, cos, tanh, reshape, transpose
from Context import using_config, no_grad
from utils import plot_dot_graph, _dot_func, _dot_var
import matplotlib.pyplot as plt

x = var([[1, 2, 3],[4,5,6]])
y = var([10,20,30])
z = x+y
print(z)

z.backward()
print(y.grad)
