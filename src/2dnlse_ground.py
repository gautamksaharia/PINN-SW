import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(123)

# ================================
# 1. Problem setup
# ================================
x_min, x_max = -6, 6
y_min, y_max = -6, 6
mu = 1.0

N_r = 100
x = torch.linspace(x_min, x_max, N_r).reshape(-1, 1)
y = torch.linspace(y_min, y_max, N_r).reshape(-1, 1)

x = (torch.rand(N_r)*2*x_max -x_max).reshape(-1,1)
y = (torch.rand(N_r)*2*y_max -y_max).reshape(-1,1)

u_target = 2.2 * torch.exp(-(x**2 + y**2))  # target Gaussian

# ================================
# 2. PINN architecture
# ================================
class PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.input(x))
        x = self.act(self.hidden1(x))
        x = self.act(self.hidden2(x))
        x = self.act(self.hidden3(x))
        return self.output(x)

# ================================
# 3. PDE residual
# ================================
def pde_residual(net, x, y):
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    X = torch.cat([x, y], dim=1)
    u = net(X)

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]

    residual = u_xx + u_yy + u**3 - mu*u
    return residual, u

# ================================
# 4. Training setup
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

Net = PINN().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(Net.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=500, gamma=0.5)

x, y, u_target = x.to(device), y.to(device), u_target.to(device)

# ================================
# 5. Training - Stage 1 (Adam)
# ================================
epochs = 5000
loss_adam = []

print("Stage 1: Training with Adam optimizer...")
start_time = time.time()
for epoch in range(epochs):
    optimizer.zero_grad()
    res, u_pred = pde_residual(Net, x, y)
    L_pde = torch.mean(res**2)
    L_data = loss_fn(u_pred, u_target)
    loss = L_data

    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_adam.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch:5d}: Total Loss = {loss.item():.6e} | PDE = {L_pde.item():.2e}")

print(f"Adam stage completed in {time.time() - start_time:.2f} sec")

# ================================
# 6. Fine-tuning - Stage 2 (LBFGS)
# ================================
loss_lbfgs = []

def closure():
    optimizer_lbfgs.zero_grad()
    res, u_pred = pde_residual(Net, x, y)
    L_pde = torch.mean(res**2)
    L_data = loss_fn(u_pred, u_target)
    loss = L_pde/(max(u_pred))
    loss.backward()
    # record loss at each LBFGS iteration
    loss_lbfgs.append(loss.item())
    return loss

print("Stage 2: Fine-tuning with LBFGS...")
optimizer_lbfgs = torch.optim.LBFGS(Net.parameters(), max_iter=500, line_search_fn="strong_wolfe")
optimizer_lbfgs.step(closure)
print("LBFGS stage completed.")

# ================================
# 7. Visualization of results
# ================================
XX, YY = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
XY = np.hstack((XX.flatten()[:, None], YY.flatten()[:, None]))
XY_torch = torch.from_numpy(XY).float().to(device)

with torch.no_grad():
    U_pred = Net(XY_torch).cpu().numpy().reshape(100, 100)

U_true = 2.2 * np.exp(-(XX**2 + YY**2))
error = U_true - U_pred

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.title("Predicted $u(x,y)$")
plt.pcolormesh(XX, YY, U_pred, shading='auto')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Target Gaussian")
plt.pcolormesh(XX, YY, U_true, shading='auto')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Error (Target - Predicted)")
plt.pcolormesh(XX, YY, error, shading='auto', cmap='coolwarm')
plt.colorbar()
plt.tight_layout()
plt.show()

# ================================
# 8. Loss Curves for Both Stages
# ================================
plt.figure(figsize=(8, 5))
plt.plot(loss_adam, label="Adam Loss", color='blue')
plt.plot(np.arange(len(loss_adam), len(loss_adam)+len(loss_lbfgs)), loss_lbfgs, label="LBFGS Loss", color='orange')
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss (Adam + LBFGS)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
