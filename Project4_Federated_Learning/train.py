import torch
import torch.nn as nn
import numpy as np
import os
from model import compile_model

def federated_averaging(models):
    global_model = compile_model()
    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        global_dict[k] = torch.stack([models[i].state_dict()[k].float() for i in range(len(models))], 0).mean(0)

    global_model.load_state_dict(global_dict)
    return global_model

def train_local_model(model, data, targets, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        for i in range(len(data)):
            inputs = data[i].unsqueeze(0)
            labels = torch.tensor([targets[i]], dtype=torch.long)
            labels = labels.argmax(dim=1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return model

def train_federated(x_train, y_train, rounds=50, clients=5, epochs=3):
    global_model = compile_model()
    for round in range(rounds):
        local_models = []
        for client in range(clients):
            local_model = compile_model()
            local_model.load_state_dict(global_model.state_dict())
            local_model = train_local_model(local_model, x_train[client], y_train[client], epochs)
            local_models.append(local_model)
        
        global_model = federated_averaging(local_models)
        print(f'Round {round+1}/{rounds} completed')
    
    return global_model

# Load and prepare data
x_train = []
y_train = []

for i in range(5):
    client_data = torch.load(os.path.join('data/clients', f'{i}', 'combined.pt'))
    x_train.append(client_data['x_train'])
    y_train.append(client_data['y_train'])

# Train the federated model
global_model = train_federated(x_train, y_train)

# Save the global model
torch.save(global_model.state_dict(), 'global_model.pt')
print('Global model saved as global_model.pt')