from kan import *
import torch
torch.set_default_dtype(torch.float64)


import matplotlib.pyplot as plt



model = KAN.loadckpt('model/0.0')




model.eval()


dummy_input = torch.randn((1, 2), dtype=torch.float64)
with torch.no_grad():
    _ = model(dummy_input)

model.plot()

plt.savefig("my_kan_diagram.png", dpi=300)
plt.close()
