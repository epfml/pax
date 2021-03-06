"""
Pax adaptation of
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import pax
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.to(device)

forward = pax.functional_module(net)
params = [p.data for p in net.parameters()]

criterion = nn.CrossEntropyLoss()
optimizer = pax.functional_optimizer(optim.SGD, lr=0.05, momentum=0.9)

opt_state = optimizer.init(params)

def loss(params, inputs, labels):
    outputs = forward(params, inputs)
    return criterion(outputs, labels)

loss_and_gradient = pax.value_and_grad(loss)

torch.cuda.synchronize()
start_time = time.time_ns()

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    n = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        loss, grads = loss_and_gradient(params, inputs, labels)
        params, opt_state = optimizer.step(params, grads, opt_state)

        # print statistics
        running_loss += loss.item()
        n += 1

    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / n))

torch.cuda.synchronize()
end_time = time.time_ns()

print('Finished Training in {}s'.format((end_time - start_time) / 1e9))