import torch
import torch.nn as nn
import numpy as np
import os
import sklearn
from model import compile_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score

def evaluate_model(model, x_test, y_test):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for i in range(len(x_test)):
            inputs = x_test[i].unsqueeze(0)
            outputs = model(inputs)
	    labels = y_test[i].argmax(dim=1)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.append(predicted.item())
            y_true.append(labels.item())
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    #roc = roc_auc_score(y_true, y_pred, multi_class='ovr')
    kappa = cohen_kappa_score(y_true, y_pred)

    return accuracy, f1, kappa

# Load test data
client_data = torch.load(os.path.join('data/clients/1', 'combined.pt'))
x_test = client_data['x_test']
y_test = client_data['y_test']

# Load the global model
global_model = compile_model()
global_model.load_state_dict(torch.load('global_model.pt'))
print('Global model loaded from global_model.pt')

# Evaluate the model
accuracy, f1, kappa = evaluate_model(global_model, x_test, y_test)

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
#print(f'ROC AUC: {roc:.4f}')
print(f'Kappa: {kappa:.4f}')
