import torch
from kan import KAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Re‑instantiate your model exactly as before
model = KAN(width=[2, 5, 20], grid=10, k=5, seed=1, device=device)
model.to(device)

# 2) Pick the checkpoint path you want (e.g. epoch 10)
ckpt_path = "KAN_4/model/model_epoch_0010.pth"
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)

# 3) Re‑create your optimizer (with the same hyperparameters)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

# (Optional) If you saved optimizer.state_dict(), load it too:
# optim_state = torch.load("…/optimizer_epoch_0010.pth")
# optimizer.load_state_dict(optim_state)

# 4) Continue training from epoch 11 onward
start_epoch = 10  # since you loaded epoch 10
epochs = 25
for epoch in range(start_epoch, epochs):
    optimizer.zero_grad()
    model.train()
    # … same training loop as before …
    # torch.save(model.state_dict(), f"KAN_4/model/model_epoch_{epoch+1:04d}.pth")
