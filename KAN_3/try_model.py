import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler


test_coords_index = 0


# --- Set up device and load the trained model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

model = KAN.loadckpt('model_initial/0.0')
model.to(device)
model.eval()  # Set model to evaluation mode

# --- Fixed x coordinates (left column) as desired ---
x_values = np.array([
    -3.999999989900970832e-06,
    -3.578947371352114715e-06,
    -3.157894752803258598e-06,
    -2.736842134254402481e-06,
    -2.315789515705546364e-06,
    -1.894736897156690247e-06,
    -1.473684278607834131e-06,
    -1.052631660058978014e-06,
    -6.315790415101218969e-07,
    -2.105264229612657800e-07,
    2.105264229612657800e-07,
    6.315790415101218969e-07,
    1.052631660058978014e-06,
    1.473684278607834131e-06,
    1.894736897156690247e-06,
    2.315789515705546364e-06,
    2.736842134254402481e-06,
    3.157894752803258598e-06,
    3.578947371352114715e-06,
    3.999999989900970832e-06
])

# --- Generate the phase profile using a single input coordinate ---

# print("Loading data from 'data.npy' ...")
data = np.load('data.npy', allow_pickle=True).item()

# Extract test data
test_input = np.array(data['test_input'])
test_label = np.array(data['test_label'])
print(f"Number of test samples: {len(test_input)}\n")


# Normalize training inputs and transform test inputs with the same scaler
input_scaler = MinMaxScaler()
test_input_norm = input_scaler.transform(test_input)

# Convert test data to torch tensors
X_test = torch.tensor(test_input_norm, dtype=torch.float64).to(device)
Y_test = torch.tensor(test_label, dtype=torch.float64).to(device)


# raw_input_coord = np.array([[0.0000e+00, -3.1111e-06]])

input_coord = torch.tensor([[0.2, 0.5]], dtype=torch.float32).to(device)

with torch.no_grad():
    phase_list = model(X_test).cpu().numpy()



# --- Pair each x value with its corresponding phase value ---
phase_points = np.column_stack((x_values, phase_list))

print("Generated phase_points:")
for point in phase_points:
    print(point)
