import torch
import matplotlib.pyplot as plt
from kan import KAN  # or however you import KAN

# 1. Instantiate the model with the same architecture as your trained model
phase_dim = 20  # Example; must match the dimension you trained with
model = KAN(width=[2, 5, 10, phase_dim], grid=3, k=3, seed=42)


model_path = 'results_1/kan_model_1.pth'  # Change if yours is different
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()  # Put the model in evaluation mode

# --- 3. Pass a NON-zero input so the model’s internal stats don’t become NaN ---
dummy_input = torch.randn((1, 2), dtype=torch.float64)  # shape must match input size (2 here)
with torch.no_grad():
    _ = model(dummy_input)

# --- 4. Plot the model ---
try:
    model.plot()  # This should generate the KAN diagram
    plt.title("Trained KAN Model Diagram")
    plt.savefig("kan_diagram.png")
except Exception as e:
    print("Plotting failed:", e)
