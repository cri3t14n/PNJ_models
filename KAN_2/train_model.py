print();print()

model_nr = 1

from kan import *
import torch
torch.set_default_dtype(torch.float64)

import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt





# --- 1. Set Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)






# --- 2. Set Up Output Folder ---
output_folder = 'results_' + str(model_nr)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print(f"Outputs will be saved in '{output_folder}' folder.\n")






# --- 3. Load and Preprocess the Data ---
print("Loading data from 'data.npy' ...")
data = np.load('data.npy', allow_pickle=True).item()

# Extract training data
train_input = np.array(data['train_input'])
train_label = np.array(data['train_label'])
print(f"Number of training samples: {len(train_input)}")

# Extract test data
test_input = np.array(data['test_input'])
test_label = np.array(data['test_label'])
print(f"Number of test samples: {len(test_input)}\n")

# Normalize training inputs and transform test inputs with the same scaler
print("Normalizing training inputs ...")
input_scaler = MinMaxScaler()
train_input_norm = input_scaler.fit_transform(train_input)
test_input_norm = input_scaler.transform(test_input)

# Convert training data to torch tensors and move to device
X_train = torch.tensor(train_input_norm, dtype=torch.float64).to(device)
Y_train = torch.tensor(train_label, dtype=torch.float64).to(device)

# Convert test data to torch tensors
X_test = torch.tensor(test_input_norm, dtype=torch.float64).to(device)
Y_test = torch.tensor(test_label, dtype=torch.float64).to(device)

print("Data loaded, normalized, and converted to torch tensors.\n")








# --- 4. Define the KAN Model ---
print("Defining the Kolmogorov-Arnold Network (KAN) model ...")
phase_dim = Y_train.shape[1]   # Output dimension (e.g., ~20)
model = KAN(width=[2, 11, phase_dim], grid=3, k=3, seed=42, device=device)
model.to(device)
print("Model defined successfully.\n")








# --- 5. Define Loss Function and Optimizer ---
print("Setting up loss function (MSE) and optimizer (Adam) ...")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print("Loss function and optimizer ready.\n")







# --- 6. Training Loop ---
num_epochs = 500
print(f"Starting training for {num_epochs} epochs...")
loss_history = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    predictions = model(X_train)  # forward pass
    loss = criterion(predictions, Y_train)
    
    loss.backward()               # backprop
    optimizer.step()              # parameter update

    loss_history.append(loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

print("Training complete.\n")

# Save the trained model
model_path = os.path.join(output_folder, f"kan_model_{model_nr}.pth")
torch.save(model.state_dict(), model_path)
model.saveckpt('./model_checkpoint')
print(f"Trained model state saved to: {model_path}\n")









# --- 7. Evaluate on Training Data for Reference ---
print("Evaluating model on training data ...")
model.eval()
with torch.no_grad():
    predicted_train = model(X_train).cpu().numpy()

# --- 7.1 Plot Training Loss ---
plt.figure(figsize=(6, 4))
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss History")
plt.legend()
loss_plot_path = os.path.join(output_folder, f"loss_history_{model_nr}.png")
plt.savefig(loss_plot_path)
print(f"Loss history plot saved to: {loss_plot_path}")
plt.close()

# --- 7.2 Plot a few training samples (for quick sanity check) ---
num_train_samples_to_plot = min(10, len(X_train))
for i in range(num_train_samples_to_plot):
    plt.figure(figsize=(6, 4))
    plt.plot(Y_train[i].cpu().numpy(), label="Ground Truth")
    plt.plot(predicted_train[i], label="Predicted", linestyle="--")
    plt.xlabel("Phase Profile Index")
    plt.ylabel("Phase Value")
    plt.legend()
    plt.title(f"Training Sample {i} Comparison")
    sample_plot_path = os.path.join(output_folder, f"train_sample_{i}_comparison.png")
    plt.savefig(sample_plot_path)
    print(f"Training sample plot saved to: {sample_plot_path}")
    plt.close()










# --- 8. Evaluate on Test Data ---
print("Evaluating model on test data ...")
with torch.no_grad():
    predicted_test = model(X_test).cpu().numpy()

# --- 8.1 Plot All Test Samples ---
num_test_samples = len(X_test)
print(f"Plotting and saving {num_test_samples} test sample comparisons...")
for i in range(num_test_samples):
    plt.figure(figsize=(6, 4))
    plt.plot(Y_test[i].cpu().numpy(), label="Ground Truth")
    plt.plot(predicted_test[i], label="Predicted", linestyle="--")
    plt.xlabel("Phase Profile Index")
    plt.ylabel("Phase Value")
    plt.legend()
    plt.title(f"Test Sample {i} Comparison")
    
    sample_plot_path = os.path.join(output_folder, f"test_sample_{i}_comparison.png")
    plt.savefig(sample_plot_path)
    print(f"Test sample plot saved to: {sample_plot_path}")
    plt.close()

print("\nAll outputs saved. Process completed.")
