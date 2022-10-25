import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_SIZE = [4, 1, 1, 1, 2]


def create_rand_params(h):
    if type(h) == nn.Linear:
        h.weight.data.uniform_(0, 1)


class NN(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        if size is None:
            size = DEFAULT_SIZE

        self.size = size
        self.layers = nn.ModuleList(
            [
                nn.Linear(self.size[i - 1], self.size[i], bias=False)
                for i in range(1, len(self.size))
            ]
        ).double()
        self.d = sum(size[i] * size[i + 1] for i in range(len(size) - 1))

    def forward(self, x):
        for i in range(len(self.size) - 2):
            x = F.leaky_relu(self.layers[i](x))
        x = self.layers[-1](x)
        x = F.softmax(x, dim=-1)
        return x


class LogReg(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        if size is None:
            size = 4

        self.size = size
        self.fc1 = nn.Linear(self.size, 1, bias=True).double()

    def forward(self, x):
        x = self.fc1(x)
        x = 1 / (1 + torch.exp(-x))
        x = torch.cat((1 - x, x), dim=-1)
        return x


if __name__ == "__main__":
    from load_data import load_data

    x, y = load_data()
    model = LogReg()
    print(model(x))
