import os
import warnings
import time
import json
from kan import *
from autograd import grad
import autograd.numpy as np
from ceviche_solver import setup_simulation_parameters, create_lens, create_source, run_simulation
from helper import load_data
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs("KAN_4/output", exist_ok=True)
os.makedirs("KAN_4/model", exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_log = {
    "epoch_loss": [],
    "gradient_norms": [],
    "epoch_times": [],
    "config": {
        "lr": 0.1,
        "max_iter": 5,
        "history_size": 10,
        "epochs": 7,
        "dataset_size": None
    }
}

X_train, y_train, X_test, y_test = load_data()
training_log["config"]["dataset_size"] = len(X_train)

######################################
# Define the loss function
######################################

params = setup_simulation_parameters()
epsr = create_lens(params)

x_min, x_max = -8e-6, 8e-6
y_min, y_max = -8e-6, 1e-6

region_mask = (
    (params['X'] >= x_min) & (params['X'] < x_max) &
    (params['Y'] >= y_min) & (params['Y'] < y_max)
)

def loss_function_field_l2(phase_vals, params, epsr, E_target, region_mask):
    source = create_source(params, phase_vals)
    Ez_pred = run_simulation(params, epsr, source)
    mse = np.mean(np.abs(((Ez_pred - E_target)[region_mask])**2))
    norm_factor = np.mean(np.abs(E_target[region_mask])**2)
    print("Normed MSE:", mse / norm_factor)
    return mse / norm_factor

def grad_loss(phase_np, E_target_np):
    def loss_wrapper(phase):
        return loss_function_field_l2(
            phase_vals=phase,
            params=params,
            epsr=epsr,
            E_target=E_target_np,
            region_mask=region_mask
        )
    grad_fn = grad(loss_wrapper)
    return grad_fn(phase_np)

######################################
# Define the gradient 
######################################

class FieldL2LossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phase_tensor, E_target_tensor):
        print("->" + "   Forward pass")
        phase_np = phase_tensor.detach().cpu().numpy()
        E_target_np = E_target_tensor.detach().cpu().numpy()
        loss_val = loss_function_field_l2(phase_np, params, epsr, E_target_np, region_mask=region_mask)
        ctx.save_for_backward(phase_tensor, E_target_tensor)
        return phase_tensor.new_tensor(loss_val, requires_grad=True)
    
    @staticmethod
    def backward(ctx, grad_output):
        print("<-" + "   Backwards pass")
        phase_tensor, E_target_tensor = ctx.saved_tensors
        phase_np = phase_tensor.detach().cpu().numpy()
        E_target_np = E_target_tensor.detach().cpu().numpy()
        dldphase_np = grad_loss(phase_np, E_target_np)
        dldphase_torch = torch.tensor(dldphase_np, dtype=phase_tensor.dtype, device=phase_tensor.device)
        return grad_output * dldphase_torch, None

def field_l2_loss_torch(phase_tensor, E_target_tensor):
    return FieldL2LossFunction.apply(phase_tensor, E_target_tensor)

######################################
# Set up the model and train
######################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KAN(width=[2,2,20], grid=5, k=3, seed=1, device=device)

def train(epochs=7, lr=1e-1):
    model.train()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=5, history_size=10, line_search_fn="strong_wolfe")
    N = len(X_train)
    start_training = time.time()
    global current_grad_norm
    current_grad_norm = 0.0
    for epoch in range(epochs):
        print(f"\n\nEpoch {epoch+1}/{epochs} | Training on {N} samples")
        start_epoch = time.time()
        def closure():
            optimizer.zero_grad()
            total_loss = 0.0
            for i in range(N):
                x_sample = X_train[i].to(device)
                E_target = y_train[i].to(device)
                predicted_phase = model(x_sample.unsqueeze(0)).squeeze(0)
                loss_i = field_l2_loss_torch(predicted_phase, E_target)
                total_loss += loss_i
            total_loss = total_loss / N
            total_loss.backward()
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item()**2
            total_norm = total_norm**0.5
            global current_grad_norm
            current_grad_norm = total_norm
            return total_loss
        loss_val = optimizer.step(closure)
        epoch_time = time.time() - start_epoch
        training_log["epoch_times"].append(epoch_time)
        training_log["epoch_loss"].append(loss_val.item())
        training_log["gradient_norms"].append(current_grad_norm)
        print(f"Epoch {epoch+1}/{epochs} | Normed Loss: {loss_val.item():.6f} | Time: {epoch_time:.2f}s | Grad Norm: {current_grad_norm:.6f}")
        
        for i in range(N):
            sample_folder = os.path.join("KAN_4", "output", f"sample_{i:04d}")
            os.makedirs(sample_folder, exist_ok=True)
            with torch.no_grad():
                sample = X_train[i].to(device)
                E_target_sample = y_train[i].to(device)
                predicted_phase = model(sample.unsqueeze(0)).squeeze(0)
                phase_np = predicted_phase.detach().cpu().numpy()
                Ez_pred = run_simulation(params, epsr, create_source(params, phase_np))
                Ez_pred = Ez_pred.reshape((params['Nx_pml'], params['Ny_pml']))
                E_target_np = E_target_sample.cpu().numpy().reshape((params['Nx_pml'], params['Ny_pml']))
                extent = [params['x_coords_pml'][0]*1e6, params['x_coords_pml'][-1]*1e6,
                          params['y_coords_pml'][0]*1e6, params['y_coords_pml'][-1]*1e6]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                im1 = ax1.imshow(np.abs(Ez_pred.T), cmap='hot', interpolation='nearest', origin='lower', extent=extent)
                ax1.scatter(sample[0].item()*1e6, sample[1].item()*1e6, c='cyan', marker='x', s=100, label='Input (x,y)')
                ax1.set_title(f'Sample {i:04d} - Predicted Field\nEpoch {epoch+1}')
                ax1.legend()
                fig.colorbar(im1, ax=ax1, label='|Ez_pred|')
                im2 = ax2.imshow(np.abs(E_target_np.T), cmap='hot', interpolation='nearest', origin='lower', extent=extent)
                ax2.scatter(sample[0].item()*1e6, sample[1].item()*1e6, c='cyan', marker='x', s=100, label='Input (x,y)')
                ax2.set_title(f'Sample {i:04d} - Target Field')
                ax2.legend()
                fig.colorbar(im2, ax=ax2, label='|Ez_target|')
                plt.tight_layout()
                plot_filename = os.path.join(sample_folder, f"epoch_{epoch+1:04d}.png")
                fig.savefig(plot_filename)
                print(f"Saved comparison plot for sample {i:04d} at epoch {epoch+1} to {plot_filename}")
                phase_info = {
                    "predicted_phase": phase_np.tolist(),
                    "loss": loss_val.item()
                }
                with open(os.path.join(sample_folder, f"phase_info_epoch_{epoch+1:04d}.json"), "w") as f:
                    json.dump(phase_info, f, indent=4)
                plt.close(fig)
        torch.save(model.state_dict(), f"KAN_4/model/model_epoch_{epoch+1:04d}.pth")
    total_training_time = time.time() - start_training
    torch.save(model.state_dict(), "KAN_4/model/final_model.pth")
    training_log["total_training_time"] = total_training_time
    with open("KAN_4/model/training_log.json", "w") as f:
        json.dump(training_log, f, indent=4)
    print("Training complete on entire training set.")

if __name__ == "__main__":
    train()
