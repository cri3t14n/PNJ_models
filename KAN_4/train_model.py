import warnings
from kan import *
from autograd import grad
import autograd.numpy as np
from ceviche_solver import setup_simulation_parameters, create_lens, create_source, run_simulation
from load import load_data

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train, y_train, X_test, y_test = load_data()



######################################
# Define the loss function
######################################

params = setup_simulation_parameters()
epsr   = create_lens(params)

x_min, x_max = -8e-6, 8e-6
y_min, y_max = -8e-6, 1e-6

region_mask = (
    (params['X'] >= x_min) & (params['X'] < x_max) &
    (params['Y'] >= y_min) & (params['Y'] < y_max)
)

def loss_function_field_l2(phase_vals, params, epsr, E_target, region_mask):
    source = create_source(params, phase_vals)
    Ez_pred = run_simulation(params, epsr, source)
    difference = (Ez_pred - E_target)[region_mask]
    mse = np.mean(np.abs(difference**2))
    print("MSE:", mse)
    return mse


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
# Define the gradient of the loss function
######################################

gradient_scaling_factor = 1


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
        return grad_output * (gradient_scaling_factor * dldphase_torch), None

def field_l2_loss(phase_tensor, E_target_tensor):
    return FieldL2LossFunction.apply(phase_tensor, E_target_tensor)


######################################
# Set up the model and train
######################################


model = KAN(width=[2,2,20], grid=5, k=3, seed=1, device=device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(epochs=10, lr=1e-1):
    model.train()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    N = len(X_train)
    
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            total_loss = 0.0
            for i in range(N):
                x_sample = X_train[i].to(device)
                E_target = y_train[i].to(device)
                predicted_phase = model(x_sample.unsqueeze(0)).squeeze(0)
                loss_i = field_l2_loss(predicted_phase, E_target)
                total_loss += loss_i
            total_loss = total_loss / N
            total_loss.backward()
            return total_loss
        
        loss_val = optimizer.step(closure)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss_val.item():.6f}")
        
        with torch.no_grad():
            sample = X_train[0].to(device)
            predicted_phase = model(sample.unsqueeze(0)).squeeze(0)
            phase_np = predicted_phase.detach().cpu().numpy()
            Ez_for_plot = run_simulation(params, epsr, create_source(params, phase_np))
            Ez_for_plot = Ez_for_plot.reshape((params['Nx_pml'], params['Ny_pml']))
            Ez_masked = np.where(region_mask, Ez_for_plot, np.nan)
            plt.figure()
            plt.imshow(np.abs(Ez_masked), cmap='hot', interpolation='nearest',
                       extent=[params['x_coords_pml'][0]*1e6, params['x_coords_pml'][-1]*1e6,
                               params['y_coords_pml'][0]*1e6, params['y_coords_pml'][-1]*1e6])
            plt.title(f'Predicted Field Epoch {epoch+1}')
            plt.colorbar()
            plt.savefig(f"KAN_4/output/predicted_field_epoch_{epoch+1:04d}.png")
            plt.close()

    print("Training complete on entire training set.")

if __name__ == "__main__":
    train(epochs=10, lr=1e-1)