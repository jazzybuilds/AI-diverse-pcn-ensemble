# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:00:31 2025

@author: pfkin
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

'''
Learning a simple ANN to classify MNIST images
'''

# --- Data ---
transform = transforms.ToTensor()
train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train, batch_size=640, shuffle=True)
test_loader  = DataLoader(test, batch_size=10000, shuffle=False)

# --- Simple ANN ---
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.net(x)

model = ANN()
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

# --- Training and Testing loop ---
print("Starting training")
accuracy_history=[]
for epoch in range(10):
    for x, y in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
            
    # --- Evaluation ---
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    test_accuracy=correct/total
    accuracy_history.append(test_accuracy)            
    print(f"Epoch {epoch+1} complete. Test accuracy {test_accuracy:.2%}")
print("Finished")

plt.figure()
plt.plot(accuracy_history)
plt.title("ANN Test accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()