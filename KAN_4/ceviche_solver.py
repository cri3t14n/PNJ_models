import autograd.numpy as np
import matplotlib.pyplot as plt
from smooth_evaluator import circular_mask, build_region_cost_matrix
from autograd import grad
import ceviche

example_phase_points = np.array([
    [-3.999999989900970832e-06, 1.209787011146545410e+00],
    [-3.578947371352114715e-06, 1.141801238059997559e+00],
    [-3.157894752803258598e-06, -2.528994560241699219e+00],    
    [-2.736842134254402481e-06, 1.787886381149291992e+00],
    [-2.315789515705546364e-06, 2.461011886596679688e+00],
    [-1.894736897156690247e-06, 3.987373113632202148e-01],
    [-1.473684278607834131e-06, -1.421883106231689453e-01],
    [-1.052631660058978014e-06, -7.687633037567138672e-01],
    [-6.315790415101218969e-07, -1.557676076889038086e+00],
    [-2.105264229612657800e-07, -1.491435170173645020e+00],
    [2.105264229612657800e-07 ,-1.444230556488037109e+00],
    [6.315790415101218969e-07 ,-1.615953803062438965e+00],
    [1.052631660058978014e-06 ,-7.585178613662719727e-01],
    [1.473684278607834131e-06 ,-9.601435065269470215e-02],
    [1.894736897156690247e-06 ,4.957991838455200195e-01],
    [2.315789515705546364e-06 ,2.484201431274414062e+00],
    [2.736842134254402481e-06 ,1.921683788299560547e+00],
    [3.157894752803258598e-06 ,-2.523023605346679688e+00],
    [3.578947371352114715e-06 ,1.036071062088012695e+00],
    [3.999999989900970832e-06 ,1.189350366592407227e+00],
])

xp_fixed = np.array(example_phase_points[:, 0])  
fp_init  =  example_phase_points[:, 1] * 0

#Autograd interp
def differentiable_interpolation(x, xp_fixed, fp):
    x = np.minimum(np.maximum(x, xp_fixed[0]), xp_fixed[-1])
    indices = np.searchsorted(xp_fixed, x) - 1
    indices = np.maximum(np.minimum(indices, len(xp_fixed) - 2), 0)
    x0 = xp_fixed[indices]
    x1 = xp_fixed[indices + 1]
    y0 = fp[indices]
    y1 = fp[indices + 1]
    weights = (x - x0) / (x1 - x0)
    return y0 + weights * (y1 - y0)


def setup_simulation_parameters():
    #Modifiable Parameters
    lambda0 = 532e-9
    c = 3e8
    omega = 2 * np.pi * c / lambda0
    n_SiO2 = 1.4607
    eps_SiO2 = n_SiO2**2
    x_width = 18.0e-6 #length not including PML
    y_width = 18.0e-6
    density = 30 #resolution in pixels per um
    pml_thickness = 2e-6 #thickness of PML added to simulation in um

    #Derived Grid Parameters
    Nx = int(x_width * 1e6 * density)    #grid
    Ny = int(y_width * 1e6 * density)
    Nx_pml = Nx + 2 * int(pml_thickness * 1e6 * density)   # grid with PML
    Ny_pml = Ny + 2 * int(pml_thickness * 1e6 * density)
    dx = x_width / Nx
    NPML = [int(pml_thickness * 1e6 * density), int(pml_thickness * 1e6 * density)]
    x_coords = np.linspace(-x_width/2, x_width/2, Nx)     # coordinate system without PML
    y_coords = np.linspace(-y_width/2, y_width/2, Ny)
    x_coords_pml = np.linspace(-x_width/2 - pml_thickness, x_width/2 + pml_thickness, Nx_pml) # coordinate system with PML
    y_coords_pml = np.linspace(-y_width/2 - pml_thickness, y_width/2 + pml_thickness, Ny_pml)
    X, Y = np.meshgrid(x_coords_pml, y_coords_pml, indexing='ij')

    print("Grid Size (um):", (x_width, y_width), "| Grid Size (pxl):", (Nx, Ny), "| PML Width (um):", (pml_thickness, pml_thickness), "| NPML (pxl):", NPML, "| Resolution (pxl/um^2):", density)

    return {
        'lambda0': lambda0, 
        'omega': omega, 
        'eps_SiO2': eps_SiO2, 
        'density': density,
        'Nx': Nx, 
        'Ny': Ny, 
        'Nx_pml': Nx_pml, 
        'Ny_pml': Ny_pml, 
        'NPML': NPML, 
        'dL': dx, 
        'x_coords': x_coords, 
        'y_coords': y_coords, 
        'x_coords_pml': x_coords_pml,
        'y_coords_pml': y_coords_pml, 
        'X': np.array(X), 
        'Y': np.array(Y), 
        'k0': 2 * np.pi / lambda0
    }

def create_lens(params):
    eps_r = np.ones((params['Nx_pml'], params['Ny_pml']), dtype=complex)
    lens_x_min = -4e-6
    lens_x_max = 4e-6
    lens_y_min = 1e-6
    lens_y_max = 9e-6
    lens_mask = (params['X'] >= lens_x_min) & (params['X'] <= lens_x_max) & (params['Y'] >= lens_y_min) & (params['Y'] <= lens_y_max)
    eps_r[lens_mask] = params['eps_SiO2']
    return eps_r



def create_source(params, xp_fixed, phase_vals):
    NPML = params['NPML']
    k0 = float(params['k0'])
    interior = (slice(NPML[0], -NPML[0]), slice(NPML[1], -NPML[1]))
    interior_shape = params['X'][interior].shape 
    Nx_int, Ny_int = interior_shape 
    x_slice = params['X'][interior][:, -1]
    y_slice = params['Y'][interior][:, -1]
    left_part = np.zeros((Nx_int, Ny_int - 1), dtype=np.complex128)

    phase_profile = differentiable_interpolation(x_slice, xp_fixed, phase_vals)
    arg = k0 * y_slice + phase_profile  #This looks weird but it helped the autograd be okay with the operations
    right_column = (np.cos(arg) + 1j * np.sin(arg))[:, None] 
    interior_source = np.concatenate([left_part, right_column], axis=1)

    source = np.pad(interior_source, #no direct assignment to source[interior] because of autograd
                    pad_width=((NPML[0], NPML[0]), (NPML[1], NPML[1])),
                    mode='constant',
                    constant_values=0.0)
    return source



def run_simulation(params, epsr, source):
    simulation = ceviche.fdfd_ez(params['omega'], params['dL'], epsr, params['NPML'])
    Ez = np.array(simulation.solve(source)[2])
    return Ez/np.max(np.abs(Ez))


def loss_function_focus(phase_vals, xp_fixed, params, epsr, target_mask, alpha=1.0):
    source = create_source(params, xp_fixed, phase_vals)
    Ez = run_simulation(params, epsr, source)
    intensity = np.abs(Ez)**2
    region_intensity = intensity[target_mask]
    mean_intensity = np.mean(region_intensity)
    return -alpha * mean_intensity



def main():
    phase_vals = fp_init
    learning_rate = 5e-2
    num_iterations = 200
    params = setup_simulation_parameters()
    epsr = create_lens(params)

    x_target = 0e-6
    y_target = -5e-6
    radius   = .33e-6

    target_mask = circular_mask(params, x_target, y_target, radius)
    def wrapped_loss(phase_vals):
        return loss_function_focus(
            phase_vals, xp_fixed, params, epsr,
            target_mask, alpha=1.0
        )

    grad_loss = grad(wrapped_loss)

    for step in range(num_iterations):
        loss_val = wrapped_loss(phase_vals)
        print(f"Step {step}: Loss = {loss_val}")
        phase_grad = grad_loss(phase_vals)
        phase_vals = phase_vals - learning_rate * phase_grad
        print("phase_grad:", phase_grad)
        print("phase_vals:", phase_vals)

        if step % 10 == 0:
            source = create_source(params, xp_fixed, phase_vals)
            Ez = run_simulation(params, epsr, source)
            ax2 = ceviche.viz.abs(Ez, outline=np.real(epsr), cbar=False, cmap='RdBu')
            ax2.plot(params['x_coords_pml'], params['y_coords_pml'], 'k')
            ax2.set_title(f"Abs Ez {step}")
            plt.savefig(f'output/ez_field_{step}.png', dpi=300, bbox_inches='tight')

    source = create_source(params, xp_fixed, phase_vals)
    Ez = run_simulation(params, epsr, source)
    ax2 = ceviche.viz.abs(Ez, outline=np.real(epsr), cbar=False, cmap='RdBu')
    ax2.plot(params['x_coords_pml'], params['y_coords_pml'], 'k')
    ax2.set_title("Abs Ez")
    plt.savefig('output/ez_field.png', dpi=300, bbox_inches='tight')
    plt.show()

    np.savetxt('output/final_Ez.txt', Ez)
    np.savetxt('output/final_phase_points.txt', phase_vals)


if __name__ == "__main__":
    main()
