from kan import *  # Import everything from the kan module
import torch
import os

# Set the default torch data type and choose the device.
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# --- Re-create the Model with the Same Architecture ---
phase_dim = 20  # Replace with your actual output dimension
model = KAN(width=[2, 10, 20, 40, phase_dim], grid=3, k=3, seed=42, device=device)
model.to(device)
print("Model instantiated and moved to device.")

# --- Load the Saved Weights from the Checkpoint ---
checkpoint_path = 'results_2/kan_model_2.pth'  # Replace with your actual checkpoint file
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from: {checkpoint_path}")
else:
    print(f"Checkpoint file not found: {checkpoint_path}")



import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

input_scaler = MinMaxScaler()
data = np.load('data.npy', allow_pickle=True).item()
train_input = np.array(data['train_input'])
train_input_norm = input_scaler.fit_transform(train_input)
X_train = torch.tensor(train_input_norm, dtype=torch.float64).to(device)


# Run a dummy forward pass so the model "sees" some data (if not already done)
sample_input = X_train[0:1]
_ = model(sample_input)

# Try plotting the model (this depends on your model having a plot() method)
try:
    model.plot()
    plt.savefig("model_plot.png")
    print("Model plot saved as 'model_plot.png'")
except Exception as e:
    print("Plotting failed:", e)