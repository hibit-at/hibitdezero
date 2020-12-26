import numpy as np
from Variable import Variable, numerical_diff, var
from Function import Function, goldstein
from Calculations import square, exp, add, mul, my_sin
from Context import using_config, no_grad
from utils import plot_dot_graph, _dot_func, _dot_var

x = Variable(np.array(np.pi/4))
y = my_sin(x,threshold=1e-150)
y.backward()
print(y.data)
print(x.grad)

plot_dot_graph(y)