import numpy as np
from dezero.core import *
from dezero import functions as F
from dezero import layers as L
from dezero import models as M
from dezero import optimizer as O
from dezero import datasets as D
import matplotlib.pyplot as plt
import math

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = D.Spiral(train=True)
test_set = D.Spiral(train=False)
train_loader = D.DataLoader(train_set, batch_size)
test_loader = D.DataLoader(test_set, batch_size, shuffle=True)

model = M.MLP((hidden_size, 10))
optimizer = O.SGD(lr).setup(model)

test_loss_graph = []
train_loss_graph = []

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch: {}'.format(epoch+1))
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    train_loss_graph.append(sum_loss / len(train_set))

    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('test loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(test_set), sum_acc / len(test_set)))
    
    test_loss_graph.append(sum_loss / len(test_set))

plt.plot(np.arange(max_epoch),train_loss_graph)
plt.plot(np.arange(max_epoch),test_loss_graph)
plt.show()
