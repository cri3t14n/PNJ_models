import matplotlib
matplotlib.use('Agg')  # Use a backend that doesn't require a display



import sys
import types
import scipy.sparse.linalg as spla
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
from sklearn.preprocessing import MinMaxScaler

from kan import KAN



# --- Optimized Solver ---
class OptimizedSolver:
    def __init__(self, A, **kwargs):
        self.factorized = spla.splu(A.tocsc())
    def factor(self):
        pass
    def solve(self, b):
        return self.factorized.solve(b)
    def __call__(self, b):
        return self.solve(b)
    def clear(self):
        del self.factorized
def optimized_pardisoSolver(A, b=None, **kwargs):
    solver = OptimizedSolver(A, **kwargs)
    if b is None:
        return solver
    else:
        return solver(b)
    
# --- Replace pyMKL with optimized solver ---
optimized_pyMKL = types.ModuleType("pyMKL")
optimized_pyMKL.pardisoSolver = optimized_pardisoSolver
sys.modules["pyMKL"] = optimized_pyMKL
from fdfdpy import Simulation


#--- Better PML ---
poly_order = 4
sig_max_val = 30
from fdfdpy.constants import EPSILON_0, ETA_0
def stronger_sig_w(l, dw, m=poly_order, lnR=-10):  
    sig_max = sig_max_val
    return sig_max*(l/dw)**m

import fdfdpy.pml as pml_module
pml_module.sig_w = stronger_sig_w
print("PML SETTINGS | poly_order:", poly_order, "| SigMax:", sig_max_val)



# ------------- Set up device and load the trained model -------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

model = KAN.loadckpt('model_initial/0.0')
model.to(device)
model.eval()  # Set model to evaluation mode

# ------------- Fixed x coordinates (left column) as desired -------------
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

# ------------- Generate the phase profile using a single input coordinate -------------
raw_input_coord = np.array([[0.0000e+00, -3.1111e-06]])

# Normalize the coordinate using the same scaler from training:
input_scaler = MinMaxScaler()
normalized_coord = input_scaler.fit_transform(raw_input_coord)

# Convert to a torch tensor (make sure the dtype matches what you used during training)
input_coord = torch.tensor(normalized_coord, dtype=torch.float32).to(device)

with torch.no_grad():
    # The model should output a list (or 1D tensor) of 20 phase values.
    phase_list = model(input_coord)

# Convert to a NumPy array and squeeze to remove extra dimensions.
phase_list = phase_list.cpu().numpy().squeeze()

print("Phase list shape:", phase_list.shape)

# Pair each x value (which should remain in physical units) with its corresponding phase value
# (Assuming x_values is defined with your fixed x coordinates in physical units)
phase_points = np.column_stack((x_values, phase_list))

print("Generated phase_points:")
for point in phase_points:
    print(point)



# ----- Simulation Parameters -----
lambda0 = 532e-9           # wavelength: 532 nm
c = 3e8                    # speed of light in vacuum
omega = 2 * np.pi * c / lambda0  # angular frequency
L0 = 1e-6                  # length scale: 1 µm

# ----- Grid Parameters -----
x_width = 18
y_width = 24
density = 40
pml_x = 3
pml_y = 3
 

# ----- Create Coordinate System -----
grid_size = (x_width*density, y_width*density)
NPML = [pml_x*density, pml_y*density]  
print("Grid Size (um):", (x_width, y_width), "| Grid Size (pxl):", grid_size, "| PML Width (um):", (pml_x, pml_y), "| NPML (pxl):", NPML, "| Resolution (pxl/um^2):", density)
nx, ny = grid_size
x_coords = np.linspace(-x_width/2, x_width/2, grid_size[0])
y_coords = np.linspace(-y_width/2, y_width/2, grid_size[1])
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
x_slice = X[NPML[0]:-NPML[0], NPML[1]:-NPML[1]][:, -1]
y_slice = Y[NPML[0]:-NPML[0], NPML[1]:-NPML[1]][:, -1]

# ----- Define the Background and Lens -----
eps_r = np.ones(grid_size, dtype=complex)
n_SiO2 = 1.4607
eps_SiO2 = n_SiO2**2      
lens_center = (0, 5)        
lens_half_side = 4      
lens_x_min = lens_center[0] - lens_half_side
lens_x_max = lens_center[0] + lens_half_side
lens_y_min = lens_center[1] - lens_half_side
lens_y_max = lens_center[1] + lens_half_side
lens_mask = (X >= lens_x_min) & (X <= lens_x_max) & (Y >= lens_y_min) & (Y <= lens_y_max)
eps_r[lens_mask] = eps_SiO2


# ----- Set Up the Source -----
k0 = 2 * np.pi / lambda0    
sim = Simulation(omega, eps_r, x_coords[1]-x_coords[0], NPML, 'Ez', L0)
sim.src = sim.src.astype(np.complex128) 
source_field = np.zeros((nx - 2 * NPML[0], ny - 2 * NPML[1]), dtype=np.complex128)


# ----- Set up the Phase -----
def create_phase_profile(phase_points):
    phase_x = phase_points[:, 0] 
    phase_vals = phase_points[:, 1]
    phase_profile = np.interp(
        x_slice*1e-6, 
        phase_x, 
        phase_vals, 
        left=phase_vals[0], 
        right=phase_vals[-1]
        )
    return phase_profile


scale = .25
# ----- Calculate Field Profile -----
def calc_field_profile(phase_profile):
    source_field[:, -1] =  scale * np.exp(1j*(k0*y_slice + phase_profile))
    sim.src[NPML[0]:-NPML[0], NPML[1]:-NPML[1]] = source_field
    print("Starting to solve fields...")
    fields = sim.solve_fields()
    Ez = fields[2] 
    Ez = Ez[NPML[0]:-NPML[0],NPML[1]:-NPML[1]] 
    return np.abs(Ez) 

# ----- Calculate Field Profile -----
phase_profile = create_phase_profile(phase_points)
Ez = calc_field_profile(phase_profile)





# ----- Compute Error -----
x_limit = 2    
y_limit_top = 9
y_limit_bot = 0
print(f"Checking error in x ∈ [{-(x_width/2)+(pml_x + x_limit)}, {(x_width/2)-(pml_x + x_limit)}] µm, "
      f"y ∈ [{-(y_width/2)+(pml_y + y_limit_bot)}, {(y_width/2)-(pml_y + y_limit_top)}] µm")
Ez_restricted = Ez[
    density*x_limit : -density*x_limit,
    density*y_limit_bot : -density*y_limit_top]
x_restricted = x_coords[(NPML[0] + density*x_limit) : -(NPML[0] + density*x_limit)]
y_restricted = y_coords[(NPML[1] + density*y_limit_bot) : -(NPML[1] + density*y_limit_top)]
X_grid, Y_grid = np.meshgrid(x_restricted, y_restricted, indexing='ij')
grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T


# ----- Plot Error -----
plt.figure(figsize=(14, 5))

# Plot FDFD solution
plt.imshow(
    Ez_restricted.T,
    origin='lower',
    extent=(x_restricted[0], x_restricted[-1], y_restricted[0], y_restricted[-1]),
    cmap='inferno',
    interpolation='bilinear')
plt.title("FDFD Simulation")
plt.xlabel("x [µm]")
plt.ylabel("y [µm]")
plt.colorbar()


plt.savefig("my_field_plot.png", dpi=300)  # Save the figure
print("Plot saved as my_field_plot.png")
