import numpy as np
import cupy as cp
from dezero.core import *
from dezero import functions as F
from dezero import functions_conv as FC
from dezero import layers as L
from dezero import models as M
from dezero import optimizer as O
from dezero import datasets as D
from dezero import cuda
from dezero import utils
import matplotlib.pyplot as plt
import math
import os

N, C, H, W = 1, 5, 15, 15
OC, (KH, KW) = 8, (3, 3)

x = Variable(np.random.randn(N,C,H,W))
x.to_gpu()
W = var(np.random.randn(OC,C,KH,KW))
print(x.shape)
y = FC.conv2d_simple(x, W, b=None, stride=1,pad=1)
y.backward()

print(x.shape)
print(x.grad.shape)

print(x)
print(y)