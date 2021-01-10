import numpy as np
from dezero.core import *
from dezero import functions as F
from dezero import layers as L
from dezero import models as M
from dezero import optimizer as O
from dezero import datasets as D
import matplotlib.pyplot as plt
import math

# Hyperparameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

x, t = dezero.datasets.get_spiral(train=True)
model = M.MLP((hidden_size, 3))
optimizer = O.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

loss_graph = []

for epoch in range(max_epoch):
    # Shuffle index for data
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

    # Print loss every epoch
    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))
    loss_graph.append(loss)

x = np.arange(len(loss_graph))
plt.plot(x,loss_graph)
plt.show()