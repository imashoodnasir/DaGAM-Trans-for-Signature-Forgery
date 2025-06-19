import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

def train(model, dataloader, epochs=10):
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
