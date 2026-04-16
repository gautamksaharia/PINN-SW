import numpy as np

# =========================
# PARAMETERS
# =========================
Lt, Lx = 10.0, 10.0
Nt, Nx = 2**7, 2**7

p = 1.2
epsn = 0.0

errormax = 1e-8
errorCG = 1e-2
c = 5.0

# =========================
# GRID SETUP
# =========================
t = np.linspace(-Lt/2, Lt/2, Nt, endpoint=False)
x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)

T, X = np.meshgrid(t, x)

# =========================
# FOURIER MODES
# =========================
kt = np.fft.fftfreq(Nt, d=Lt/Nt) * 2*np.pi
kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2*np.pi

KT, KX = np.meshgrid(kt, kx)

# Linear operator in Fourier space
K2 = KT + 0.5 * KX**2 + epsn * KX**3

# Preconditioner
fftM = c + K2**2

# =========================
# INITIAL CONDITION (Peregrine)
# =========================
U = 1 - 4*(1 + 2j*T)/(1 + 4*X**2 + 4*T**2)

# =========================
# HELPER FUNCTIONS
# =========================
def spectral_op(U):
    return np.fft.ifft2(-K2 * np.fft.fft2(U))

def residual(U):
    return spectral_op(U) + (np.abs(U)**(2*p))*U - U

def dN(U, D):
    return (p+1)*(np.abs(U)**(2*p))*D + \
           p*(U**2)*(np.abs(U)**(2*p-2))*np.conj(D)

def L1(U, D):
    return spectral_op(D) - D + dN(U, D)

# =========================
# NEWTON-CG LOOP
# =========================
ncg = 0
ITER = 20000
flag = True

while flag and ncg <= ITER:

    # Residual
    L0U = residual(U)
    err = np.max(np.abs(L0U))
    print(f"Residual: {err:.3e}")

    if err < errormax:
        break

    # Define operators
    def L1A(D):
        return L1(U, D)

    # RHS
    R = -L1A(L0U)

    DU = np.zeros_like(U)

    # Preconditioned residual
    MinvR = np.fft.ifft2(np.fft.fft2(R) / fftM)
    R2new = np.sum(np.conj(R) * MinvR)
    R20 = R2new

    P = MinvR.copy()

    # =========================
    # CONJUGATE GRADIENT LOOP
    # =========================
    while np.abs(R2new) > np.abs(R20)*(errorCG**2) and flag:

        L1P = L1(U, P)
        LP = L1A(L1P)

        denom = np.sum(np.real(np.conj(P) * LP))
        if denom == 0:
            break

        a = R2new / denom

        DU = DU + a * P
        R = R - a * LP

        MinvR = np.fft.ifft2(np.fft.fft2(R) / fftM)

        R2old = R2new
        R2new = np.sum(np.real(np.conj(R) * MinvR))

        b = R2new / R2old
        P = MinvR + b * P

        ncg += 1

    # =========================
    # NEWTON UPDATE
    # =========================
    U = U + DU

# =========================
# OUTPUT
# =========================
U_abs = np.abs(U)

print("Max amplitude:", np.max(U_abs))




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assume U, x, t already computed
U_abs = np.abs(U)

X_plot, T_plot = np.meshgrid(t, x)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    T_plot, X_plot, U_abs,
    linewidth=0,
    antialiased=True
)

ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('|U(x,t)|')

plt.title('Rogue Wave (Peregrine Soliton)')
plt.tight_layout()
plt.show()


plt.figure(figsize=(7,5))

levels = np.linspace(U_abs.min(), U_abs.max(), 40)

plt.contourf(
    t, x, U_abs,
    levels=levels
)

plt.colorbar(label='|U(x,t)|')
plt.xlabel('t')
plt.ylabel('x')
plt.title('Rogue Wave Contour Plot')

plt.tight_layout()
plt.show()


mid_t = Nt // 2

plt.figure(figsize=(6,4))
plt.plot(x, U_abs[:, mid_t])

plt.xlabel('x')
plt.ylabel('|U(x, t=0)|')
plt.title('Spatial Profile at Time t=0')

plt.grid()
plt.show()
