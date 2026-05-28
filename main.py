import torch
from config import CONFIG
from data import get_dataloaders
from model import VisionTransformer
from train import train, validate, overfit_test

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainloader, testloader = get_dataloaders(CONFIG)

# Sanity check
model = VisionTransformer(**CONFIG).to(device)
overfit_test(model, trainloader, device)

# Real training
model = VisionTransformer(**CONFIG).to(device)
train(model, trainloader, CONFIG["learning_rate"], CONFIG["epochs"], device)
validate(model, testloader, device)