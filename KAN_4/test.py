import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from kan import KAN
from ceviche_solver import setup_simulation_parameters, create_lens, create_source, run_simulation

# -------------------------------------------------
# User configuration
# -------------------------------------------------
x = 1.0384e-6  
y = -2e-6 
model_dir = "KAN_4/model"
output_dir = "output_inference"

# -------------------------------------------------
# Helper: find .pth file
# -------------------------------------------------
def find_latest_model(model_dir):
    pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in {model_dir}")
    # Prefer final_model.pth if it exists
    final_path = os.path.join(model_dir, "final_model.pth")
    if final_path in pth_files:
        return final_path
    # Otherwise, return most recently modified
    pth_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return pth_files[0]

# -------------------------------------------------
# Main inference routine
# -------------------------------------------------
def main():
    os.makedirs(output_dir, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Locate model
    model_path = find_latest_model(model_dir)
    print(f"Loading model from: {model_path}")


    params = setup_simulation_parameters()
    epsr = create_lens(params)


    model = KAN(width=[2, 5, 20], grid=10, k=5, seed=1, device=device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()


    x_scaled = x * 1e6
    y_scaled = y * 1e6
    input_tensor = torch.tensor([x_scaled, y_scaled], dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        phase_pred = model(input_tensor).squeeze(0).cpu().numpy()


    source = create_source(params, phase_pred)
    Ez_flat = run_simulation(params, epsr, source)
    Ez = Ez_flat.reshape((params["Nx_pml"], params["Ny_pml"]))
    extent = [
        params["x_coords_pml"][0] * 1e6,
        params["x_coords_pml"][-1] * 1e6,
        params["y_coords_pml"][0] * 1e6,
        params["y_coords_pml"][-1] * 1e6,
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        np.abs(Ez.T),
        cmap="hot",
        interpolation="nearest",
        origin="lower",
        extent=extent
    )

    ax.scatter(x * 1e6, y * 1e6, c="cyan", marker="x", s=100, label="Source (x, y)")
    ax.set_title(f"Predicted |Ez| Field at x={x:.2e}, y={y:.2e} m")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.legend()
    fig.colorbar(im, ax=ax, label="|Ez|")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
