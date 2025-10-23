# PINN based iterative solver
# L u = 0
# u_xx + |u|^2u + mu*u= 0, mu =-2
# u(x) = sqrt(-2 mu) sech(sqrt(-mu) x)
# u_0 = sech( x)
# domain [-20,20] , N =500



import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
torch.manual_seed(123)
from torch.optim.lr_scheduler import StepLR



x_min = -20
x_max = 20
N_r = 5000
x = torch.linspace(x_min, x_max, N_r).reshape(N_r,1)

mu = -2  #bright
u0 = 1/torch.cosh(x) # initial guess bright soliton



# [1, 32, 32,32, 32, 1]
# act func = tanh(x) 
# act func = sin(x)

class Neural_network(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super(Neural_network, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer0 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.input_layer(x))
        x = self.act(self.hidden_layer0(x))
        x = self.act(self.hidden_layer1(x))
        x = self.output_layer(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Net = Neural_network().to(device)
loss_fu = nn.MSELoss()
optim = torch.optim.Adam(Net.parameters(), lr=0.001, weight_decay=1e-3)



def ode_residual(Net, x):
    x1 = torch.autograd.Variable(x.float(), requires_grad=True).to(device)
    u = Net.forward(x1)
    u_x = torch.autograd.grad(u.sum(), x1, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x1, create_graph=True)[0]
    
    ode = u_xx  - u**3 + mu*u           
    return ode, u 


number_of_epoch =1000
loss_value = []

start_time = time.time()
for epoch in range(number_of_epoch):
  optim.zero_grad()       # make the gradient zero
  res, _ = ode_residual(Net, x)
  uu = Net.forward(x)
    
  Loss_ode = torch.mean(res**2)
  loss_guess = torch.mean((uu- u0)**2)
  
  
  total_loss = loss_guess     # total loss hard constraint
  total_loss.backward()    # computing gradients using backward propagation  dL/dw
  optim.step()             # This is equivalent to : Weight_new = weight_old - learing_rate * derivative of Loss w.r.t weight
  # Step the scheduler
  #scheduler.step()
  loss_value.append(total_loss.cpu().detach().numpy())
  with torch.autograd.no_grad():
    if epoch%20==0:
      print(f'epoch:{epoch}, loss:{total_loss.item():.8f}')
print("total time:",time.time() - start_time, "seconds")


def closure():
    optimizer_lbfgs.zero_grad()
    res, u  = ode_residual(Net, x)
    Loss_ode = torch.mean(res**2)/(max(u))
    total_loss =   Loss_ode      # total loss
    total_loss.backward(retain_graph=True)  # Keep the graph for LBFGS optimization
    return total_loss

# Initialize LBFGS optimizer
optimizer_lbfgs = torch.optim.LBFGS(Net.parameters(), max_iter=15000, line_search_fn="strong_wolfe")
# Run LBFGS optimization
optimizer_lbfgs.step(closure)


X_tensor = torch.linspace(x_min,x_max, 500).reshape(-1,1)
upred1dnlse = Net.forward(X_tensor).detach().cpu().numpy()
plt.plot(X_tensor, abs(upred1dnlse))
plt.xlabel("x")
plt.ylabel("u")
plt.title("pred")
plt.show()

xx = np.linspace(x_min,x_max,500)
utrue = np.sqrt(-2*mu)*(1/np.cosh(np.sqrt(-mu)*xx)) # bright
uguess = np.tanh(xx)
plt.plot(xx, utrue, label="true")
plt.plot(xx, uguess, label="guess")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.title("true")
plt.show()


plt.plot(X_tensor, abs(upred1dnlse), label="pred")
plt.plot(xx, abs(utrue), label="true")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.title("compare")
plt.show()




