# def train_single_sample_adam(sample_idx=20, epochs=10, lr=1e-2):
#     model.train()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Simpler optimizer
    
#     x_sample = X_train[sample_idx].to(device)
#     E_target = y_train[sample_idx].to(device)

#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         predicted_phase = model(x_sample.unsqueeze(0)).squeeze(0)
#         loss = field_l2_loss(predicted_phase, E_target)
#         loss.backward()
#         optimizer.step()

#         logging.info(f"[Single Sample] Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

#         with torch.no_grad():
#             phase_np = predicted_phase.cpu().numpy()
#             Ez_for_plot = run_simulation(params, epsr, create_source(params, phase_np))
#             Ez_for_plot = Ez_for_plot.reshape((params['Nx_pml'], params['Ny_pml']))
#             Ez_masked = np.where(region_mask, Ez_for_plot, np.nan)

#             fig, ax = plt.subplots()
#             im = ax.imshow(np.abs(Ez_masked), cmap='hot', interpolation='nearest',
#                            extent=[params['x_coords_pml'][0]*1e6, params['x_coords_pml'][-1]*1e6,
#                                    params['y_coords_pml'][0]*1e6, params['y_coords_pml'][-1]*1e6])
#             ax.set_title(f'Sample {sample_idx} | Epoch {epoch+1}')
#             fig.colorbar(im, ax=ax, label='|Ez|')

#             output_path = f"KAN_4/output/sample_{sample_idx}_epoch_{epoch+1:04d}.png"
#             fig.savefig(output_path)
#             plt.close(fig)
#             logging.info(f"Plot saved: {output_path}")

#     torch.save(model.state_dict(), f"KAN_4/model/sample_{sample_idx}_final_model.pth")
#     logging.info("Training complete on a single sample.")


#Working sigle sample 

# def train_single_sample(sample_idx=20, epochs=10, lr=1e-1):
#     model.train()
#     optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=5, history_size=10, line_search_fn="strong_wolfe")

#     x_sample = X_train[sample_idx].to(device)
#     E_target = y_train[sample_idx].to(device)
#     x_coord, y_coord = x_sample.cpu().numpy()

#     for epoch in range(epochs):
#         print(f"\n\nEpoch {epoch+1}/{epochs} | Sample {sample_idx+1}/{len(X_train)}")

#         def closure():
#             optimizer.zero_grad()
#             predicted_phase = model(x_sample.unsqueeze(0)).squeeze(0)
#             print("\nPredicted Phase:", predicted_phase)
#             loss = field_l2_loss(predicted_phase, E_target)
#             loss.backward()
#             return loss
#         loss_val = optimizer.step(closure)

#         # Plotting
#         with torch.no_grad():
#             predicted_phase = model(x_sample.unsqueeze(0)).squeeze(0)
#             phase_np = predicted_phase.detach().cpu().numpy()
#             Ez_pred = run_simulation(params, epsr, create_source(params, phase_np))
#             Ez_pred = Ez_pred.reshape((params['Nx_pml'], params['Ny_pml']))

#             E_target_np = E_target.cpu().numpy().reshape((params['Nx_pml'], params['Ny_pml']))

#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#             extent = [params['x_coords_pml'][0]*1e6, params['x_coords_pml'][-1]*1e6,
#                       params['y_coords_pml'][0]*1e6, params['y_coords_pml'][-1]*1e6]

#             # Predicted Field
#             im1 = ax1.imshow(np.abs(Ez_pred.T), cmap='hot', interpolation='nearest', origin='lower', extent=extent)
#             ax1.scatter(x_coord*1e6, y_coord*1e6, c='cyan', marker='x', s=100, label='Input (x,y)')
#             ax1.set_title(f'Predicted Field Epoch {epoch+1}')
#             ax1.legend()
#             fig.colorbar(im1, ax=ax1, label='|Ez_pred|')

#             # Target Field
#             im2 = ax2.imshow(np.abs(E_target_np.T), cmap='hot', interpolation='nearest', origin='lower', extent=extent)
#             ax2.scatter(x_coord*1e6, y_coord*1e6, c='cyan', marker='x', s=100, label='Input (x,y)')
#             ax2.set_title('Target Field')
#             ax2.legend()
#             fig.colorbar(im2, ax=ax2, label='|Ez_target|')

#             plt.tight_layout()
#             output_path = f"KAN_4/output/single_sample_comparison_epoch_{epoch+1:04d}.png"
#             fig.savefig(output_path)
#             print(f"Saved comparison plot to {output_path}")
#             plt.close(fig)

#             print(f"\n[Single Sample] Epoch {epoch+1}/{epochs} | Normed Loss: {loss_val.item():.6f}")

#     torch.save(model.state_dict(), "KAN_4/model/final_model_single_sample.pth")
#     print("Training complete on a single sample.")