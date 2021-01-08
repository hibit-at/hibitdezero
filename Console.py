import numpy as np
from dezero.core import *
from dezero import functions as F
from dezero import layers as L
from dezero import models as M
from dezero import optimizer as O
import matplotlib.pyplot as plt

model = M.MLP((10,4))

x = var([[2,-4],[3,5],[1,-3],[2,3]])
t = var([2,0,1,0])
y = model(x)
loss = F.softmax_cross_entropy_simple(y, t)
print(loss)