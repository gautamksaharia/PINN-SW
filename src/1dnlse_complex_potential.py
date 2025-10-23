# PINN based iterative solver 
# 1 D NLSE complex potential
# L u = 0
# u_xx - g|u|^2u + mu*u - V(x)u = 0
# V = V0 sech^2(x) + i W0 sech(x)tanh(x)
#mu =1, V0 = -1 , g = -1, W0 = -1
# u_0 = sech(x)e^(ix)
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

mu = -1

# self defocusing
g = 1 
V0 = -3
W0 = -1

# self focusing
#g = -1 
#V0 = -1
#W0 = -1

p0 = 1/torch.cosh(x)*torch.cos(x)  # real part
q0 = 1/torch.cosh(x)*torch.sin(x)  # imag part


# [1, 32, 32,32, 32, 2]
# act func = tanh(x) 
# act func = sin(x)

class Neural_network(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=2):
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
    p = Net.forward(x1)[:, 0:1]
    q =Net.forward(x1)[:, 1:2]

    Vr = V0*(1/torch.cosh(x1))**2
    Vm = W0* (1/torch.cosh(x1))*torch.tanh(x1)

    p_x = torch.autograd.grad(p.sum(), x1, create_graph=True)[0]
    p_xx = torch.autograd.grad(p_x.sum(), x1, create_graph=True)[0]

    q_x = torch.autograd.grad(q.sum(), x1, create_graph=True)[0]
    q_xx = torch.autograd.grad(q_x.sum(), x1, create_graph=True)[0]
    
    ode1 = p_xx  -g*(p**2 + q**2)*p + mu*p - Vr*p  + Vm * q
    ode2 = q_xx  -g*(p**2 + q**2)*q + mu*q - Vr*q  - Vm * p

            
    return ode1,ode2, p, q  


number_of_epoch =1000
loss_value = []


start_time = time.time()
for epoch in range(number_of_epoch):
  optim.zero_grad()       # make the gradient zero
  res1, res2, _, _ = ode_residual(Net, x)
  pp = Net.forward(x)[:, 0:1]
  qq = Net.forward(x)[:, 1:2]
    
  Loss_ode = torch.mean(res1**2) + torch.mean(res2**2) 
  loss_guess = torch.mean((pp- p0)**2) + torch.mean((qq- q0)**2)
    
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
    res1, res2, pp,qq = ode_residual(Net, x)
    uu = pp + 1j*qq
    Loss_ode = (torch.mean(res1**2) + torch.mean(res2**2))/(max(abs(uu)))
    total_loss =   Loss_ode      # total loss
    total_loss.backward(retain_graph=True)  # Keep the graph for LBFGS optimization
    return total_loss

# Initialize LBFGS optimizer
optimizer_lbfgs = torch.optim.LBFGS(Net.parameters(), max_iter=100, line_search_fn="strong_wolfe")
# Run LBFGS optimization
optimizer_lbfgs.step(closure)

X_tensor = torch.linspace(x_min,x_max, 500).reshape(-1,1)
ppred1= Net.forward(X_tensor)[:, 0:1].detach().cpu().numpy()
qpred1= Net.forward(X_tensor)[:, 1:2].detach().cpu().numpy()
plt.plot(X_tensor, np.sqrt(ppred1**2 + qpred1**2) )  
plt.xlabel("x")
plt.ylabel("u")
plt.title("pred")
plt.show()

xx = np.linspace(x_min,x_max,500)
utrue = np.sqrt(- (2 + V0 + (W0**2) / 9)/ g )*(1/np.cosh(xx))*np.exp(-1j*W0/3 * np.arctan(np.sinh(xx))) # self focusing case
uguess = (1/np.cosh(xx))*np.exp(1j*xx)
plt.plot(xx, np.abs(utrue), label="true")
plt.plot(xx, np.abs(uguess), label="guess")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.title("true")
plt.show()

plt.plot(X_tensor, np.sqrt(ppred1**2 + qpred1**2), label="pred")
plt.plot(xx, np.abs(utrue), label="true")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.title("compare")
plt.show()


