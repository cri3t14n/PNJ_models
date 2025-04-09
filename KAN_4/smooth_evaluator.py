import autograd.numpy as np

def q_factor_at_point(Ez, X, Y, x_target, y_target, radius=0.3e-6):
    dist = np.sqrt( (X - x_target)**2 + (Y - y_target)**2 )
    region_mask = dist <= radius
    region_vals = Ez[region_mask]
    if region_vals.size == 0:
        return 0.0
    return np.mean(region_vals)


def circular_mask(params, x_center, y_center, radius):
    dx = params['X'] - x_center
    dy = params['Y'] - y_center
    return (dx**2 + dy**2) <= radius**2

def build_region_cost_matrix(params, mask):
    region_ix = np.where(mask.ravel())[0]  # returns array of indices
    Xsub = params['X'].ravel()[region_ix]
    Ysub = params['Y'].ravel()[region_ix]
    N = len(Xsub)
    Mregion = np.zeros((N, N))  # <-- Autograd zeros
    for i in range(N):
        dx = Xsub[i] - Xsub
        dy = Ysub[i] - Ysub
        Mregion[i, :] = np.sqrt(dx*dx + dy*dy)
    return Mregion, region_ix





