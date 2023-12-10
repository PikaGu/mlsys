import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import needle.data as data
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    return nn.Sequential(
        nn.Residual(modules),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    loss_fn = nn.SoftmaxLoss()
    total_loss = 0
    count = 0
    err = 0
    num_samples = 0
    for batch in dataloader:
        X, y = batch
        X = X.reshape((X.shape[0], -1))
        out = model(X)
        loss = loss_fn(out, y)
        total_loss += loss.cached_data
        count += 1
        err += np.sum(np.argmax(out.cached_data, axis=1) != y.cached_data)
        num_samples += X.shape[0]
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()
    print(err, num_samples, total_loss, count)
    return err / num_samples, total_loss / count
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = data.MNISTDataset(
        data_dir + "/train-images-idx3-ubyte.gz",
        data_dir + "/train-labels-idx1-ubyte.gz"
    )
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataset = data.MNISTDataset(
        data_dir + "/t10k-images-idx3-ubyte.gz",
        data_dir + "/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size
    )
    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        train_err, train_loss = epoch(train_dataloader, model, opt)
        test_err, test_loss = epoch(test_dataloader, model)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(ep, train_loss, train_err, test_loss, test_err))
    return tuple([train_err, train_loss, test_err, test_loss])
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
