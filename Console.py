import numpy as np
from dezero.core import *
from dezero import functions as F
import matplotlib.pyplot as plt
import dezero.layers as L

model = L.Layer()
model.l1 = L.Linear(5)
model.l2 = L.Linear(2)

print(model)

def predict(model, x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y

for p in model.params():
    print(p)

model.cleargrad()