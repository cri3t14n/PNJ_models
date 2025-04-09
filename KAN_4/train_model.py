import torch
import numpy as np
from autograd import grad
import torch.nn as nn
import torch.optim as optim
from ceviche_solver import wrapped_loss

grad_loss = grad(wrapped_loss)

print("Loading data from 'data.npy' ...")
data = np.load('data.npy', allow_pickle=True).item()

train_input = np.array(data['train_input'])   
train_label = np.array(data['train_label'])   
test_input  = np.array(data['test_input'])    
test_label  = np.array(data['test_label'])    

print(f"Number of training samples: {len(train_input)}")
print(f"Number of test samples: {len(test_input)}\n")

X_train = torch.tensor(train_input, dtype=torch.float32)
y_train = torch.tensor(train_label, dtype=torch.float32)
X_test  = torch.tensor(test_input,  dtype=torch.float32)
y_test  = torch.tensor(test_label,  dtype=torch.float32)


class SimLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predicted_phase_tensor, x_target_tensor, y_target_tensor):
        phase_vals = predicted_phase_tensor.detach().cpu().numpy()
        x_target_np = float(x_target_tensor.item())
        y_target_np = float(y_target_tensor.item())
        loss_val = wrapped_loss(phase_vals, x_target_np, y_target_np) # sim
        ctx.save_for_backward(predicted_phase_tensor, x_target_tensor, y_target_tensor) # for backward pass
        return torch.tensor(loss_val, dtype=predicted_phase_tensor.dtype,
                            device=predicted_phase_tensor.device)

    @staticmethod
    def backward(ctx, grad_output):
        predicted_phase_tensor, x_target_tensor, y_target_tensor = ctx.saved_tensors
        phase_vals = predicted_phase_tensor.detach().cpu().numpy()
        x_target_np = float(x_target_tensor.item())
        y_target_np = float(y_target_tensor.item())

        # get gradient wrt phase
        phase_grad_np = grad_loss(phase_vals, x_target_np, y_target_np)
        print(f"Gradient wrt phase: {phase_grad_np}")
        phase_grad_torch = torch.tensor(phase_grad_np,
                                        dtype=predicted_phase_tensor.dtype,
                                        device=grad_output.device)
        return grad_output * phase_grad_torch, None, None

def sim_loss(predicted_phase_tensor, x_target_tensor, y_target_tensor):
    return SimLossFunction.apply(predicted_phase_tensor, x_target_tensor, y_target_tensor)


class PhasePredictor(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)  


def train_with_simulation(X_train, epochs=10, lr=1e-3):
    model = PhasePredictor(in_dim=2, hidden=32, out_dim=20)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        sim_loss_vals = []

        for i in range(len(X_train)):
            optimizer.zero_grad()
            phase_i = model(X_train[i].unsqueeze(0)).squeeze(0)
            print(f"Phase for sample {i}: {phase_i}")
            x_t = X_train[i, 0]
            y_t = X_train[i, 1]
            print(f"Computing Sim for coordinates: ({x_t}, {y_t})")
            sim_val_i = sim_loss(phase_i, x_t, y_t)
            print(f"Sim loss for sample {i}: {sim_val_i.item()}")
            sim_loss_vals.append(sim_val_i)
            sim_val_i.backward()
            optimizer.step()

        # Average across the batch
        final_loss = torch.mean(torch.stack(sim_loss_vals))
        print(f"Epoch {epoch+1}/{epochs} | Simulation Loss: {final_loss.item():.6f}")

    return model

if __name__ == "__main__":
    print("Training model with simulation loss ...")
    trained_model = train_with_simulation(X_train, epochs=10, lr=1e-3)
    trained_model.eval()
    with torch.no_grad():
        preds = trained_model(X_train[:5])
        print("\nPredicted phases for first 5 training samples:")
        print(preds)
