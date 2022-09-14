# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.opflow import AerPauliExpectation
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %%
qi = QuantumInstance(Aer.get_backend("qasm_simulator"), shots=1024)

# %%
training_data = datasets.MNIST(
    download=True,
    root="data",
    train=True,
    transform=transforms.ToTensor(),
)

test_data = datasets.MNIST(
    download=True,
    root="data",
    train=False,
    transform=transforms.ToTensor(),
)

# remove data with labels other than 0 and 1
training_data.data = training_data.data[training_data.targets < 2]
training_data.targets = training_data.targets[training_data.targets < 2]
test_data.data = test_data.data[test_data.targets < 2]
test_data.targets = test_data.targets[test_data.targets < 2]


# only select first 512 samples
training_data.data = training_data.data[:512]
training_data.targets = training_data.targets[:512]
test_data.data = test_data.data[:512]
test_data.targets = test_data.targets[:512]
train_loader = DataLoader(
    training_data,
    batch_size=64,
    shuffle=True,
)
test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=True,
)


# %%
# plots six first images in a 2x3 grid
fig, axes = plt.subplots(2, 3)
for i, ax in enumerate(axes.flatten()):
    ax.imshow(training_data[i][0].squeeze(), cmap="gray_r")
    ax.set_title(training_data[i][1])
    ax.axis("off")

# %%
feature_map = ZZFeatureMap(feature_dimension=2, entanglement="linear")
ansatz = RealAmplitudes(2, reps=1, entanglement="linear")
qnn = TwoLayerQNN(
    2,
    feature_map,
    ansatz,
    input_gradients=True,
    exp_val=AerPauliExpectation(),
    quantum_instance=qi,
)


# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 2)
        self.qnn = TorchConnector(qnn)
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.qnn(x)
        x = self.fc3(x)
        return torch.cat((x, 1 - x), -1)


# %%
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.NLLLoss()

epochs = 20
loss_list = []

model.train()
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Calculating loss
        loss = loss_func(output, target)
        # Backward pass
        loss.backward()
        # Optimize the weights
        optimizer.step()

        total_loss.append(loss.item())
    loss_list.append(sum(total_loss) / len(total_loss))
    print(
        "Training [{:.0f}%]\tLoss: {:.4f}".format(
            100.0 * (epoch + 1) / epochs, loss_list[-1]
        )
    )


# Loss convergence plot
plt.plot(loss_list)
plt.title("Hybrid NN Training Convergence")
plt.xlabel("Training Iterations")
plt.ylabel("Neg. Log Likelihood Loss")
plt.show()

# %%
torch.save(model.state_dict(), "model_mnist.pt")

# %%
model.eval()
with torch.no_grad():

    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = loss_func(output, target)
        total_loss.append(loss.item())

    print(
        "Performance on test data:\n\t"
        "Loss: {:.4f}\n\tAccuracy: {:.1f}%".format(
            sum(total_loss) / len(total_loss),
            (correct / len(test_loader) / 64) * 100,
        )
    )

# %%
n_samples_show = 5
count = 0
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(15, 5))
model.eval()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if count == n_samples_show:
            break
        output = model(data[0:1])
        if len(output.shape) == 1:
            output = output.reshape(1, *output.shape)

        pred = output.argmax(dim=1, keepdim=True)
        axes[count].imshow(data[0].numpy().squeeze(), cmap="gray_r")

        axes[count].set_xticks([])
        axes[count].set_yticks([])

        if pred.item() == 0:
            axes[count].set_title("Predicted 0")
        elif pred.item() == 1:
            axes[count].set_title("Predicted 1")

        count += 1

# %%
