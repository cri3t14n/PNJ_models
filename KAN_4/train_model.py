print(); print()

from kan import *
import torch
from torch.autograd import Function
torch.set_default_dtype(torch.float64)

import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Import the wrapped loss and its gradient from your simulation module
from ceviche_solver import wrapped_loss, grad_loss

# =============================================================================
# Define the Custom Autograd Function for Simulation Loss
# =============================================================================
class SimLossFunction(Function):
    @staticmethod
    def forward(ctx, phase_tensor):
        # Convert torch tensor to numpy array (detached and on CPU)
        phase_vals = phase_tensor.detach().cpu().numpy()
        
        # Compute the simulation loss using the wrapped_loss function
        loss_val = wrapped_loss(phase_vals)
        
        # Save phase_vals for the backward pass
        ctx.phase_vals = phase_vals
        
        # Return the loss as a torch tensor (keeping same dtype and device)
        return torch.tensor(loss_val, dtype=phase_tensor.dtype, device=phase_tensor.device)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved phase values from the forward pass
        phase_vals = ctx.phase_vals
        
        # Compute gradient of the simulation loss with respect to phase_vals
        phase_grad_np = grad_loss(phase_vals)
        
        # Convert the numpy gradient back to a torch tensor
        phase_grad = torch.tensor(phase_grad_np, dtype=torch.float64, device=grad_output.device)
        
        # Respect the chain rule by multiplying with grad_output
        return grad_output * phase_grad

# =============================================================================
# 1. Set Device
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# =============================================================================
# 2. Set Up Folders for Results, Errors, and Outputs
# =============================================================================
results_folder = 'results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

error_folder = 'errors'
if not os.path.exists(error_folder):
    os.makedirs(error_folder)

output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# =============================================================================
# 3. Load and Preprocess the Data
# =============================================================================
def normalize_data(data, data_min=10, data_max=100, feature_range=(0, 1)):
    a, b = feature_range
    normalized = (data - data_min) / (data_max - data_min)
    scaled = normalized * (b - a) + a
    return scaled

# Example usage of normalization function:
data_example = np.array([20, 50, 100])
normalized_data_example = normalize_data(data_example)
print("Example normalized data:", normalized_data_example)

print("Loading data from 'data.npy' ...")
data = np.load('data.npy', allow_pickle=True).item()

# Extract training and test data
train_input = np.array(data['train_input'])
train_label = np.array(data['train_label'])
print(f"Number of training samples: {len(train_input)}")

test_input = np.array(data['test_input'])
test_label = np.array(data['test_label'])
print(f"Number of test samples: {len(test_input)}\n")

# Normalize training inputs and apply same transformation to test inputs
print("Normalizing training inputs ...")
input_scaler = MinMaxScaler()
train_input_norm = input_scaler.fit_transform(train_input)
test_input_norm = input_scaler.transform(test_input)

# Convert data to torch tensors and move to the proper device
X_train = torch.tensor(train_input_norm, dtype=torch.float64).to(device)
Y_train = torch.tensor(train_label, dtype=torch.float64).to(device)

X_test = torch.tensor(test_input_norm, dtype=torch.float64).to(device)
Y_test = torch.tensor(test_label, dtype=torch.float64).to(device)

print("Data loaded, normalized, and converted to torch tensors.\n")

# =============================================================================
# 4. Define the KAN Model
# =============================================================================
print("Defining the Kolmogorov-Arnold Network (KAN) model ...")
# Here phase_dim is the number of phase points output by the model (e.g., ~20)
phase_dim = Y_train.shape[1]
model = KAN(width=[2, 10, phase_dim], grid=3, k=3, seed=42, auto_save=True, device=device)
model.to(device)
print("Model defined successfully.\n")

# =============================================================================
# 5. Set Up the Optimizer
# =============================================================================
print("Setting up optimizer (Adam) ...")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print("Optimizer ready.\n")

# =============================================================================
# 6. Training Loop using the Simulation Loss
# =============================================================================
num_epochs = 100
print(f"Starting training for {num_epochs} epochs using simulation loss...")
loss_history = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Get the phase prediction from the KAN model
    phase_pred = model(X_train)
    
    # Compute the simulation loss using our custom autograd function
    sim_loss = SimLossFunction.apply(phase_pred)
    
    # Backward pass using the gradient from the simulation loss
    sim_loss.backward()
    optimizer.step()

    loss_history.append(sim_loss.item())
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Simulation Loss: {sim_loss.item():.6f}")

print("Training complete.\n")

# =============================================================================
# 7. Plot and Save the Training Loss History
# =============================================================================
plt.figure(figsize=(6, 4))
plt.plot(loss_history, label="Simulation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss History (Simulation Loss)")
plt.legend()
loss_plot_path = os.path.join(results_folder, "loss_history.png")
plt.savefig(loss_plot_path)
print(f"Loss history plot saved to: {loss_plot_path}")
plt.close()

# =============================================================================
# 8. Evaluate on Test Data
# =============================================================================
print("Evaluating model on test data using simulation loss ...")
model.eval()
with torch.no_grad():
    phase_test = model(X_test)
    # Compute simulation loss on test predictions
    test_sim_loss = SimLossFunction.apply(phase_test)
    
print(f"Test simulation loss: {test_sim_loss.item():.6f}")

# =============================================================================
# (Optional) Save Test Predictions
# =============================================================================
test_pred_np = phase_test.cpu().numpy()
np.save(os.path.join(results_folder, "test_predictions.npy"), test_pred_np)
print(f"Test predictions saved to: {os.path.join(results_folder, 'test_predictions.npy')}")

print("\nAll outputs saved. Process completed.")
