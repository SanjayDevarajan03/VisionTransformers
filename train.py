import math
import torch
import torch.nn as nn
import torch.optim as optim


def train(model, trainloader, learning_rate, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_steps = len(trainloader) * epochs
    warmup_steps = total_steps // 10

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if i % 100 == 0:
                print(f'Epoch {epoch+1}, Step {i}, Loss: {loss.item():.3f}, Avg: {running_loss/(i+1):.3f}, LR: {scheduler.get_last_lr()[0]:.6f}')


def validate(model, testloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')
    model.train()


def overfit_test(model, trainloader, device):
    images, labels = next(iter(trainloader))
    images, labels = images.to(device), labels.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for step in range(200):
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(images), labels)
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(f'Step {step}, Loss: {loss.item():.3f}')