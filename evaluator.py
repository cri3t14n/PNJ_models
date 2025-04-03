import numpy as np
import matplotlib.pyplot as plt

#read example to np array
data = np.genfromtxt('/Users/tobiasmikkelsen/Desktop/My_projects/Mirzah/Ez.csv', delimiter=',')
Ez = data.T

def find_peak(Ez, h_sens = .5, v_sens = .5):
    peak_idx_full = np.unravel_index(np.argmax(Ez), Ez.shape)
    peak_value = Ez[peak_idx_full]
    # Horizontal FWHM calculation
    row = peak_idx_full[0]
    line_data = Ez[row, :]
    left_candidates = np.where(line_data[:peak_idx_full[1]] < peak_value * h_sens)[0]
    left_idx = left_candidates[-1] if left_candidates.size > 0 else 0
    right_candidates = np.where(line_data[peak_idx_full[1]:] < peak_value * h_sens)[0]
    right_idx = peak_idx_full[1] + (right_candidates[0] if right_candidates.size > 0 else len(line_data) - peak_idx_full[1])
    fwhm = right_idx - left_idx
    # Vertical FWHM calculation
    col = peak_idx_full[1]
    col_data = Ez[:, col]
    top_candidates = np.where(col_data[:peak_idx_full[0]] < peak_value * v_sens)[0]
    top_idx = top_candidates[-1] if top_candidates.size > 0 else 0
    bottom_candidates = np.where(col_data[peak_idx_full[0]:] < peak_value * v_sens)[0]
    bottom_idx = peak_idx_full[0] + (bottom_candidates[0] if bottom_candidates.size > 0 else len(col_data) - peak_idx_full[0])
    flhm = bottom_idx - top_idx
    return peak_value, peak_idx_full, fwhm, flhm


def q_factor1(Ez, peak_idx_full, fwhm, flhm):
    I_max = Ez[peak_idx_full]
    L_eff = flhm 
    Q = (I_max * L_eff) / fwhm
    return Q


def IntGFactor(Ez, peak_idx_full, fwhm, flhm, surround_radius=100):
    cy, cx = peak_idx_full
    # PNJ region
    left = max(0, cx - fwhm // 2)
    right = min(Ez.shape[1], cx + fwhm // 2 + 1)
    top = max(0, cy - flhm // 2)
    bottom = min(Ez.shape[0], cy + flhm // 2 + 1)
    PNJ_avg = np.sum(Ez[top:bottom, left:right]) / ((bottom - top) * (right - left))
    # Surrounding area
    left_sur = max(0, cx - surround_radius)
    right_sur = min(Ez.shape[1], cx + surround_radius + 1)
    top_sur = max(0, cy - surround_radius)
    bottom_sur = min(Ez.shape[0], cy + surround_radius + 1)
    Surround_avg = np.sum(Ez[top_sur:bottom_sur, left_sur:right_sur]) / ((bottom_sur - top_sur) * (right_sur - left_sur))
    return PNJ_avg / Surround_avg





peak_value, peak_idx_full, fwhm, flhm = find_peak(Ez)
Q = IntGFactor(Ez, peak_idx_full, fwhm, flhm)

print("Peak Value:", peak_value)
print("Peak Index:", peak_idx_full)
print("FWHM_Hor:", fwhm)
print("FWHM_Vert:", flhm)
print("Q-Factor:", Q)


plt.figure(figsize=(8, 6))
plt.imshow(Ez, cmap='inferno', origin='lower')
plt.colorbar(label='Intensity')
plt.scatter([peak_idx_full[1]], [peak_idx_full[0]], color='cyan', marker='x', s=100, label='Detected Peak')
plt.title("Detected Nanojet Peak (within ROI)")
plt.xlabel("x index")
plt.ylabel("y index")
plt.legend()
plt.show()