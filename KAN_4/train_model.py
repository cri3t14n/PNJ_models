import os
import warnings
import torch
import matplotlib.pyplot as plt
from kan import *
from autograd import grad
from helper import load_data
from ceviche_solver import (setup_simulation_parameters, create_lens, create_source, run_simulation)
from torch.utils.data import DataLoader 
import autograd.numpy as np


# DO NOT CHANGE THE ORDER OF THE IMPORTS, ONE OF THE IMPORTS USES REGULAR NUMPY 

warnings.filterwarnings("ignore")
os.makedirs("KAN_4/output", exist_ok=True)
os.makedirs("KAN_4/model", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, y_train, X_test, y_test = load_data()
X_train_scaled = X_train * 1e6 #scale as the input to the model is too small and does not distinguish between the values
X_test_scaled = X_test * 1e6

# -------------------------------------------------
# Simulation set‑up
# -------------------------------------------------
params = setup_simulation_parameters()
epsr = create_lens(params)

x_min, x_max = -6e-6, 6e-6
y_min, y_max = -8e-6, 1e-6
region_mask = ( (params["X"] >= x_min) & (params["X"] < x_max) & (params["Y"] >= y_min) & (params["Y"] < y_max))


def get_target_index(params, x_target, y_target):
    diff = np.abs(params["X"] - x_target) + np.abs(params["Y"] - y_target)
    idx = np.unravel_index(np.argmin(diff), diff.shape)
    return idx


# -------------------------------------------------
# Loss + custom autograd 
# -------------------------------------------------

def loss_function_field_l2(phase_vals, params, epsr, E_target, region_mask, target_coord, target_alpha=0.6):
    source = create_source(params, phase_vals)
    Ez_pred = run_simulation(params, epsr, source)
    mse = np.mean(np.abs(((Ez_pred - E_target)[region_mask]) ** 2))
    norm_factor = np.mean(np.abs(E_target[region_mask]) ** 2)
    target_strength = np.abs(Ez_pred[get_target_index(params, target_coord[0], target_coord[1])])
    target_loss = (target_strength - 1) ** 2
    total_loss = mse / norm_factor + target_alpha * target_loss #norm of the gradient of the field times target alpha
    return total_loss


def grad_loss(phase_np, E_target_np, target_coord):
    def loss_wrapper(phase):
        return loss_function_field_l2(
            phase_vals=phase,
            params=params,
            epsr=epsr,
            E_target=E_target_np,
            region_mask=region_mask,
            target_coord=target_coord,
        )
    grad_fn = grad(loss_wrapper)
    return grad_fn(phase_np)


class FieldL2LossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phase_tensor, E_target_tensor, target_coord_tensor):
        phase_np = phase_tensor.detach().cpu().numpy()
        E_target_np = E_target_tensor.detach().cpu().numpy()
        target_coord = (target_coord_tensor[0].item(), target_coord_tensor[1].item())
        loss_val = loss_function_field_l2(phase_np, params, epsr, E_target_np, region_mask, target_coord)
        ctx.save_for_backward(phase_tensor, E_target_tensor, target_coord_tensor)
        return phase_tensor.new_tensor(loss_val, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        phase_tensor, E_target_tensor, target_coord_tensor = ctx.saved_tensors
        phase_np = phase_tensor.detach().cpu().numpy()
        E_target_np = E_target_tensor.detach().cpu().numpy()
        target_coord = (target_coord_tensor[0].item(), target_coord_tensor[1].item())
        dldphase_np = grad_loss(phase_np, E_target_np, target_coord)
        dldphase_torch = torch.tensor(dldphase_np, dtype=phase_tensor.dtype, device=phase_tensor.device)
        return grad_output * dldphase_torch, None, None


def field_l2_loss(phase_tensor, E_target_tensor, target_coord_tensor):
    return FieldL2LossFunction.apply(
        phase_tensor, E_target_tensor, target_coord_tensor
    )


# -------------------------------------------------
# Model
# -------------------------------------------------

BATCH = 16  
model = KAN(width=[2, 5, 20], grid=10, k=5, seed=1, device=device) #model = KAN(width=[2, 5, 20], grid=10, k=5, seed=1, device=device) from meeting
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)

# -------------------------------------------------
# Training loop
# -------------------------------------------------

def train(epochs=15):
    print("\nTraining on device:", device)
    N = len(X_train)
    loader = DataLoader(range(N), batch_size=BATCH, shuffle=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        for batch_idx in loader:

            x_batch = X_train_scaled[batch_idx].to(device) 
            E_target_batch = y_train[batch_idx].to(device)
            phase_pred = model(x_batch) 

            loss_accum = 0.0
            for j, idx in enumerate(batch_idx):
                x_sample = x_batch[j]
                E_target = E_target_batch[j]
                x_target = x_sample[0].item() * 1e-6
                y_target = x_sample[1].item() * 1e-6
                target_coord_tensor = torch.tensor([x_target, y_target], device=device, dtype=torch.float32)
                loss_ij = field_l2_loss(phase_pred[j], E_target, target_coord_tensor)
                loss_accum = loss_accum + loss_ij

            loss = loss_accum / len(batch_idx)
            epoch_loss += loss.item() * len(batch_idx)
            print(f"Batch {batch_idx[0]}: loss = {loss.item():.6f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: mean loss = {epoch_loss / N:.6f}")

        with torch.no_grad():
            for i in range(min(5, N)):
                sample_folder = os.path.join("KAN_4", "output", f"sample_{i:04d}")
                os.makedirs(sample_folder, exist_ok=True)

                sample = X_train_scaled[i].to(device)
                E_target_sample = y_train[i].to(device)
                predicted_phase = model(sample.unsqueeze(0)).squeeze(0)
                phase_np = predicted_phase.detach().cpu().numpy()
                Ez_pred = run_simulation(params, epsr, create_source(params, phase_np))
                Ez_pred = Ez_pred.reshape((params["Nx_pml"], params["Ny_pml"]))
                E_target_np = (E_target_sample.cpu() .numpy() .reshape((params["Nx_pml"], params["Ny_pml"])))
                extent = [params["x_coords_pml"][0] * 1e6, params["x_coords_pml"][-1] * 1e6, params["y_coords_pml"][0] * 1e6, params["y_coords_pml"][-1] * 1e6]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                im1 = ax1.imshow( np.abs(Ez_pred.T), cmap="hot", interpolation="nearest", origin="lower", extent=extent)
                ax1.scatter(sample[0].item(), sample[1].item(), c="cyan", marker="x", s=100, label="Input (x,y)")
                ax1.set_title(f"Sample {i:04d} - Predicted Field\nEpoch {epoch + 1}")
                ax1.legend()
                fig.colorbar(im1, ax=ax1, label="|Ez_pred|")

                im2 = ax2.imshow( np.abs(E_target_np.T), cmap="hot", interpolation="nearest", origin="lower", extent=extent)
                ax2.scatter( sample[0].item(), sample[1].item(), c="cyan", marker="x", s=100, label="Input (x,y)")
                ax2.set_title(f"Sample {i:04d} - Target Field")
                ax2.legend()
                fig.colorbar(im2, ax=ax2, label="|Ez_target|")
                plt.tight_layout()
                plot_filename = os.path.join(sample_folder, f"epoch_{epoch + 1:04d}.png")
                fig.savefig(plot_filename)
                plt.close(fig)

        # Save checkpoint each epoch
        torch.save(model.state_dict(), f"KAN_4/model/model_epoch_{epoch + 1:04d}.pth")

    # Final model
    torch.save(model.state_dict(), "KAN_4/model/final_model.pth")
    print("Training complete on entire training set.")


if __name__ == "__main__":
    train()
