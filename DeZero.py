import numpy as np
from Variable import Variable, numerical_diff, var
from Function import Function, goldstein, rosenbrock
from Calculations import square, exp, add, mul, my_sin, sin, cos
from Context import using_config, no_grad
from utils import plot_dot_graph, _dot_func, _dot_var
import matplotlib.pyplot as plt

x = var(np.linspace(-7,7,200))
y = sin(x)
y.backward(create_graph=True)

logs = [y.data.flatten()]

for i in range(3):
    logs.append(x.grad.data.flatten())
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

labels = ["y=sin(x)","y'","y''","y'''"]
for i,v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc='lower right')
plt.show()