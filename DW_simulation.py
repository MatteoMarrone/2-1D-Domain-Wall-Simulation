import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

import time
import os
import re
from datetime import datetime
import pytz

#WELCOME TO THE SIMULATION OF 2+1D COSMOLOGICAL DOMAIN WALLS.
#---------------------------------------------------------------------
#We consider a scalar field theory over a curved background ds2 = a^2(eta)(-deta^2 + dx^2 + dy^2), with a potential
#of the type V(phi) = 1/8 (phi^2 - 1)^2, Z2 symmetric with two stable vacua phi=+-1

#THE EQUATION WE WILL SOLVE IS: #phi.. = -alpha*H*dphi/dt + Laplacian(phi) - 0.5*phi*(phi**2-1)*a(eta)^beta
#WHERE a IS THE SCALE FACTOR AND H IS THE CONFORMAL HUBBLE PARAMETER, H = a'/a, where a' = da/deta
#eta is the conformal time.
#The parameters alpha and beta are such that alpha=1 and beta=2 in the physical scheme, i.e the true equation of motion
#but alpha=2, beta=0 in the "fattening" scheme, where the DW width remains constant on comoving coordinates. Useful for simulations

pi = np.pi

plot_initial_field = False #we plot the initial field phi
computation = True #simulation
anim_phi = True #animation of field phi
anim_V = True #animation of potential V
anim_gradsqphi = True #animation of (nabla phi)^2
plot_ratios = True #we plot the fraction of positive and negative values of phi over eta
misc_data = True #we save some data in a file
plot_mean_rms = True #we plot the mean value of phi and rms of phi over eta and save the png file
plot_mean_pmsigma = True
plot_DWlength = True

show_plots = False #we show the plots i.e plt.show()

save_step = 10 #we save field phi values every "save_step" steps

N = 500
L = 100.
h = L/N
print(f"grid spacing: {h:.3f}")
deta = 0.2*h
print(f"deta: {deta:.3f}")

nmax = L/(2.*pi)
# print("nmax instability is: ", nmax)
# This is related with the maximum number of unstable modes when one considers at initial time a field configuration with 0 average and certain rms 
# (see below the section "initial condition for phi")

#Damping term in the evolution
Gamma = 0.
etamin, etamax = 10, 15 #Gamma !=0 between etamin and etamax

#cosmological parameters
eta_initial = 1.
p = 2./3. #a(t) propto t^p. for radiation in 2+1 we have p=2/3
# c = p/(1-p) #a(eta) propto eta^c
c = 1.

print(f"scale factor grows as eta ^ {c:.2f}")
H_initial = c/eta_initial

#fattening parameters, in 2+1
idx = 1 #0 = physical, 1 = fattening scheme
print("--We are considering physical scheme") if idx==0 else print("--We are considering fattening scheme")
alpha = [1, 2]
beta = [2, 0]

x = np.linspace(0,L,N,endpoint=False)
y = np.linspace(0,L,N,endpoint=False)
X,Y = np.meshgrid(x,y)

#final time of evolution given by resolution conditions. We want to have enough grid resolution for
#the width of a domain wall, which has a physical size typically of m^-1 = 1, nonetheless in the coordinates we are
#working with, the conformal coordinates, the physical sizes grows over time as a(eta), i.e X_phys = a(eta)*X
#Since we build the grid on the comoving coordinates X and Y, the final time of evolution is given by the condition
#that dX_phys(final) = 1 = a(eta_final)dX, where dX = h = L/N

if idx==0: #physical scheme
    eta_final = min( h**(-1./c) if c!=0 else 0.5*L, c*L if c !=0 else 1e7)
else: #fattening
    eta_final = 0.5*c*L if c != 0 else 0.5*L #cL is given by the time where the lattice = one Hubble patch

print(f"Final eta is: {eta_final:.1f}")

total_steps = (eta_final - eta_initial)/deta
print(f"Total steps of simulation will be: {total_steps:.0f}")


filename_data = "DW_data(eta)(ratiop)(ratiom)(mean)(rms)(sigmaphi)(DWlength_L2).dat"
#------------------------------------------------------------------------

def gradient_squared(f):
    """
    Computes (df/dx)^2 + (df/dy)^2
    """
    # Central differences with periodic BC
    df_dx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2. * h)
    df_dy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2. * h)
    
    return df_dx**2 + df_dy**2

def dw_length(phi):
    """
    Calculates the total length of domain walls from a certain field configuration.
    We use eq. 8 of Oliveira, Martins and Avelino, 2004
    """
    dphi_dx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * h)
    dphi_dy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * h)
    
    # Magnitude of gradient
    grad_phi_mag = np.sqrt(dphi_dx**2 + dphi_dy**2)
    
    # Denominator in the formula
    grad_phi_sum_abs = np.abs(dphi_dx) + np.abs(dphi_dy)
    
    # Avoid division by zero
    grad_phi_sum_abs[grad_phi_sum_abs == 0] = 1e-6
    
    # Compute the weight field
    weight_field = grad_phi_mag / grad_phi_sum_abs

    # Detect sign changes between neighbors (horizontal and vertical)
    sign_change_h = phi[:, :-1] * phi[:, 1:] < 0
    sign_change_v = phi[:-1, :] * phi[1:, :] < 0

    # Average the weight at the two points forming the link
    weight_h = 0.5 * (weight_field[:, :-1] + weight_field[:, 1:])
    weight_v = 0.5 * (weight_field[:-1, :] + weight_field[1:, :])

    # Sum weighted contributions only where sign change occurs
    total = np.sum(weight_h[sign_change_h]) + np.sum(weight_v[sign_change_v])
    
    # Multiply by grid spacing
    delta_L = h  
    return delta_L * total

def momentum_conservation(eta, phi_old, phi_new):
    df_deta = (phi_new - phi_old)/deta
    df_dx = (np.roll(phi_new, -1, axis=0) - np.roll(phi_new, 1, axis=0)) / (2. * h)
    df_dy = (np.roll(phi_new, -1, axis=1) - np.roll(phi_new, 1, axis=1)) / (2. * h)

    a = (eta/eta_initial)**c

    Px = np.sum( df_dx*df_deta*a )
    Py = np.sum( df_dy*df_deta*a )

    grad = (df_dx)**2 + (df_dy)**2
    pot = (1./8.)*(phi_new**2-1)**2
    H = np.sum(0.5*df_deta**2.+0.5*grad+pot) / N**2 #total energy per site

    return Px,Py,H

def laplacian(f):
    # Periodic neighbors
    f_ip = np.roll(f, -1, axis=0)  # i+1
    f_im = np.roll(f, 1, axis=0)   # i-1
    f_jp = np.roll(f, -1, axis=1)  # j+1
    f_jm = np.roll(f, 1, axis=1)   # j-1

    lap = (f_ip - 2*f + f_im) / h**2 + (f_jp - 2*f + f_jm) / h**2
    return lap

def f(eta,phi,v):
    """
    RHS. It is the part of the equation phi.. = f(phi)
    """
    lapphi = laplacian(phi)

    #in case we consider a damping term Gamma
    g = Gamma if etamin <= eta <= etamax else 0

    H = H_initial * (eta_initial / eta)  # Hubble parameter, H = a' / a
    a = (eta / eta_initial)**c  # Scale factor

    return lapphi - 0.5*phi*(phi**2-1)*a**beta[idx] - g*v - alpha[idx]*H*v

def rk4_step(eta,phi, v):
    """
    Solves the system:
        dphi/deta = v
        dv/deta = f(phi)
    using RK4.
    """

    k1_phi = v
    k1_v = f(eta,phi,v)

    k2_phi = v + 0.5 * deta * k1_v
    k2_v = f(eta+deta/2.,phi + 0.5 * deta * k1_phi, k2_phi)

    k3_phi = v + 0.5 * deta * k2_v
    k3_v = f(eta+deta/2.,phi + 0.5 * deta * k2_phi, k3_phi)

    k4_phi = v + deta * k3_v
    k4_v = f(eta+deta,phi + deta * k3_phi, k4_phi)

    phi_new = phi + (deta/6.0) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
    v_new = v + (deta/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return phi_new, v_new

#------------------------------------------------------------------

#INITIAL CONDITION FOR PHI
# print("We are computing the initial profile...")

# Number of modes
M_max = int(np.ceil(nmax/2.)) #how many cos modes we want for the initial condition
print("Number of modes is: ", M_max)

# Random phases
theta_x = 2 * pi * np.random.rand(M_max, M_max)
theta_y = 2 * pi * np.random.rand(M_max, M_max)

phi = np.zeros_like(X)
print("We are computing the initial field...")
for m in range(1, M_max + 1):
    for n in range(1, M_max + 1):
        phi += (1./M_max**2)*np.cos(2 * np.pi * m * X / L + theta_x[m-1, n-1]) * np.cos(2 * np.pi * n * Y / L + theta_y[m-1, n-1])

# Enforce the mean we want initially
desired_mean = 0.
phi += desired_mean-np.mean(phi)

current_rms = np.sqrt(np.mean(phi**2))
print(f"initial rms: {current_rms:.3f}. ", f"initial average: {desired_mean:.1f}")

print("Initial field computed.")

#INITIAL CONDITION FOR DPHI/DT
# dtheta_x = np.random.rand(M_max, M_max)
# dtheta_y = np.random.rand(M_max, M_max)

# dtphi = np.zeros_like(X)
# for m in range(1, M_max + 1):
#     for n in range(1, M_max + 1):
#         dtphi += (1./(L*M_max**2))*np.sin(2 * np.pi * m * X / L + dtheta_x[m-1, n-1]) * np.sin(2 * np.pi * n * Y / L + dtheta_y[m-1, n-1])

# desired_mean_dtphi = 0.
# dtphi += desired_mean_dtphi-np.mean(dtphi)

# current_rms_dtphi = np.sqrt(np.mean(dtphi**2))
dtphi = np.zeros_like(phi)

print("Initial dtphi computed.")

#FOLDERS AND FILES
if Gamma == 0:
    folder_name = f"DW(N={N})(L={L})(deta={deta:.3f})(c={c:.2f})(Mmax={M_max})(avgphi={desired_mean:.1f})(rmsphi={current_rms:.3f})(avgdtphi={np.mean(dtphi):.3f})(rmsdtphi={np.sqrt(np.mean(dtphi**2)):.4f})(g={Gamma})"
else:
    folder_name = f"DW(N={N})(L={L})(deta={deta:.3f})(c={c:.2f})(Mmax={M_max})(avgphi={desired_mean:.1f})(rmsphi={current_rms:.3f})(avgdtphi={np.mean(dtphi):.3f})(rmsdtphi={np.sqrt(np.mean(dtphi**2)):.4f}) - (g={Gamma})(etamin={etamin})(etamax={etamax})"

if computation:
    # Save initial field
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(f"{folder_name}/snapshots_videos", exist_ok=True)
    np.save(f"{folder_name}/snapshots_videos/phi_step_000000.npy", phi)
else:
    pass

#Plot initial field
if plot_initial_field == True:
    plt.figure(figsize=(6, 5))
    plt.imshow(phi.T, extent=(0, L, 0, L), origin='lower', cmap='cividis')
    plt.colorbar(label='φ')
    plt.title('Field φ at INITIAL Time')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show() if show_plots else None 
else:
    pass

phi_new = np.ones_like(phi)
tol=1e-4
step=0
eta = eta_initial

start_time = time.time()

if computation == True:
    print("We are computing the time evolution...")
    #exact time of start of simulation
    tz = pytz.timezone("Europe/Madrid") #change depending on your location
    ctime = datetime.now(tz)
    print(f"Computation started at: {ctime.strftime('%H:%M:%S')}")

    if misc_data == True:
        contador_plus = np.sum(phi >= 0)
        contador_minus = np.sum(phi < 0)
        fil_data = open(f"{folder_name}/{filename_data}", "w")
        fil_data.write(f"{eta_initial:.3f} {contador_plus/N**2} {contador_minus/N**2} {desired_mean} {current_rms} {current_rms} {dw_length(phi)/L**2} \n")

    else:
        pass

    while eta < eta_final:
        step += 1
        eta += deta
        phi_new, dtphi_new = rk4_step(eta,phi, dtphi)

        mean = np.mean(phi_new)
        rms = np.sqrt(np.mean(phi_new**2))
        sigma = np.sqrt(np.mean((phi_new - mean)**2))
        dwl = dw_length(phi_new)/L**2

        if misc_data == True:
            contador_plus = np.sum(phi_new >= 0)
            contador_minus = np.sum(phi_new < 0)
            fil_data.write(f"{eta:.3f} {contador_plus/N**2} {contador_minus/N**2} {mean} {rms} {sigma} {dwl}\n")
        else:
            pass

        phi, dtphi = phi_new, dtphi_new

        if step % save_step == 0: #we save every save_step steps
            invH = eta/c if c!=0 else 0
            print(f"step: {step}, eta: {eta:.2f}, invH: {invH:.2f}, mean: {mean:.3f}, rms: {rms:.3f}, sigma: {sigma:.3f}, dwl_L2: {dwl:.3f}")
            filename = f"{folder_name}/snapshots_videos/phi_step_{step:06d}.npy"
            np.save(filename, phi)


    print(f"Total time of computation (s): {time.time() - start_time:.2f}")

# --------------------------
if anim_phi==True:
    # Folder with saved snapshots
    folder = f"{folder_name}/snapshots_videos"

    # List all .npy files
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]

    # Extract step numbers from filenames like 'phi_step_001000.npy'
    def extract_step(filename):
        match = re.search(r'phi_step_(\d+)\.npy', filename)
        return int(match.group(1)) if match else -1

    # Sort files by step number
    files_sorted = sorted(files, key=extract_step)

    # Load first file to get grid shape
    first_phi = np.load(os.path.join(folder, files_sorted[0]),allow_pickle=True)

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(first_phi.T, origin='lower', extent=(0,L,0,L), cmap='viridis', vmin=-1.2, vmax=1.2)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Step {extract_step(files_sorted[0])}")

    #we create an square of radius H^-1
    square = patches.Rectangle((0,0), 0, 0, edgecolor='white', facecolor='none', lw=1.5)
    ax.add_patch(square)

    def update(frame):
        #time
        eta = eta_initial + float(extract_step(files_sorted[frame])) * deta

        phi = np.load(os.path.join(folder, files_sorted[frame]), allow_pickle=True)

        im.set_data(phi.T)
        ax.set_title(rf"Field $\phi$ for $L={L}m^{{-1}}$ and $N={N}$. $\eta= {eta:.2f}m^{{-1}}$")
        ax.set_xlabel(r"$x/m^{-1}$")
        ax.set_ylabel(r"$y/m^{-1}$")

        # Update square size
        H = H_initial * (eta_initial / eta)  # Hubble parameter, H
        size = 1./H if H!=0 else 0
        square.set_width(size)
        square.set_height(size)

        return [im, ax.title, square]  # Return both image and title for blitting

    ani = animation.FuncAnimation(fig, update, frames=len(files_sorted), interval=100, blit=False)

    plt.show() if show_plots else None 
    print("saving phi-field file mp4...")
    ani.save(os.path.join(folder, 'phi_evolution.mp4'), writer='ffmpeg', fps=10)
    print("file mp4 saved.")
    plt.close(fig)
else:
    pass

if anim_V==True:
    # Folder with saved snapshots
    folder = f"{folder_name}/snapshots_videos"

    # List all .npy files
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]

    # Extract step numbers from filenames like 'phi_step_001000.npy'
    def extract_step(filename):
        match = re.search(r'phi_step_(\d+)\.npy', filename)
        return int(match.group(1)) if match else -1

    # Sort files by step number
    files_sorted = sorted(files, key=extract_step)

    # Load first file to get grid shape
    first_phi = np.load(os.path.join(folder, files_sorted[0]),allow_pickle=True)
    first_V = (first_phi**2-1)**2

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(first_V.T, origin='lower', extent=(0,L,0,L), cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Step {extract_step(files_sorted[0])}")

    #we create an square of radius H^-1
    square = patches.Rectangle((0,0), 0, 0, edgecolor='white', facecolor='none', lw=1.5)
    ax.add_patch(square)

    def update(frame):
        #time
        eta = eta_initial + float(extract_step(files_sorted[frame])) * deta

        phi = np.load(os.path.join(folder, files_sorted[frame]), allow_pickle=True)
        V = (phi**2 - 1)**2

        im.set_data(V.T)
        ax.set_title(rf"Potential $V/V_0$ for $L={L}m^{{-1}}$ and $N={N}$. $\eta= {eta:.2f}m^{{-1}}$")
        ax.set_xlabel(r"$x/m^{-1}$")
        ax.set_ylabel(r"$y/m^{-1}$")

        # Update square size
        H = H_initial * (eta_initial / eta)  # Hubble parameter, H
        size = 1./H if H!=0 else 0
        square.set_width(size)
        square.set_height(size)

        return [im, ax.title, square]  # Return both image and title for blitting

    ani = animation.FuncAnimation(fig, update, frames=len(files_sorted), interval=100, blit=False)

    plt.show() if show_plots else None 
    print("saving V-potential file mp4...")
    ani.save(os.path.join(folder, '!V_evolution.mp4'), writer='ffmpeg', fps=10)
    print("file mp4 saved.")
    plt.close(fig)
else:
    pass

if anim_gradsqphi == True:
    # Folder with saved snapshots
    folder = f"{folder_name}/snapshots_videos"

    # List all .npy files
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]

    # Extract step numbers from filenames like 'phi_step_001000.npy'
    def extract_step(filename):
        match = re.search(r'phi_step_(\d+)\.npy', filename)
        return int(match.group(1)) if match else -1

    # Sort files by step number
    files_sorted = sorted(files, key=extract_step)

    # Load first file to get grid shape
    first_phi = np.load(os.path.join(folder, files_sorted[0]),allow_pickle=True)
    first_V = (first_phi**2-1)**2 / 8.

    first_gradsqphi = gradient_squared(phi)

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(first_gradsqphi.T, origin='lower', extent=(0,L,0,L), cmap='viridis', vmin=0, vmax=0.5)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Step {extract_step(files_sorted[0])}")

    #we create an square of radius H^-1
    square = patches.Rectangle((0,0), 0, 0, edgecolor='white', facecolor='none', lw=1.5)
    ax.add_patch(square)

    def update(frame):
        #time
        eta = eta_initial + float(extract_step(files_sorted[frame])) * deta

        phi = np.load(os.path.join(folder, files_sorted[frame]), allow_pickle=True)
        V = (phi**2 - 1)**2 / 8.

        a = (eta / eta_initial)**c  # Scale factor
        gradsqphi = gradient_squared(phi) if idx==1 else gradient_squared(phi)/a**2

        im.set_data(gradsqphi.T)
        if idx==1:
            ax.set_title(rf"$(\nabla\phi)^2$ for $L={L}m^{{-1}}$ and $N={N}$. $\eta= {eta:.2f}m^{{-1}}$")
        else:
            ax.set_title(rf"$(\nabla\phi)^2/a^2$ for $L={L}m^{{-1}}$ and $N={N}$. $\eta= {eta:.2f}m^{{-1}}$")

        ax.set_xlabel(r"$x/m^{-1}$")
        ax.set_ylabel(r"$y/m^{-1}$")

        # Update square size
        H = H_initial * (eta_initial / eta)  # Hubble parameter, H
        size = 1./H if H!=0 else 0
        square.set_width(size)
        square.set_height(size)

        return [im, ax.title, square]  # Return both image and title for blitting

    ani = animation.FuncAnimation(fig, update, frames=len(files_sorted), interval=100, blit=False)

    plt.show() if show_plots else None 
    print("saving -gradient squared phi- file mp4...")
    ani.save(os.path.join(folder, '!gradsqphi_evolution.mp4'), writer='ffmpeg', fps=10)
    print("file mp4 saved.")
    plt.close(fig)
else:
    pass


if plot_ratios == True:
    with open(f"{folder_name}/{filename_data}") as fil_ratios:
        lines = fil_ratios.readlines()
        eta = np.array([line.split()[0] for line in lines], dtype=float)
        plus = np.array([line.split()[1] for line in lines], dtype=float)
        minus = np.array([line.split()[2] for line in lines], dtype=float)

    plt.plot(eta, plus, label='Plus')
    plt.plot(eta, minus, label='Minus')
    # plt.plot(eta, plus+minus,label='Plus + Minus', linestyle='--',color='black')
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"Fraction")
    plt.legend()
    
    plt.savefig(f"{folder_name}/ratiospm_eta.png")
    print("ratios plus minus png file saved.")
    plt.show() if show_plots else None 
    plt.clf()

else:
    pass

if plot_mean_rms == True:
    with open(f"{folder_name}/{filename_data}") as fil_mean:
        lines = fil_mean.readlines()
        eta = np.array([line.split()[0] for line in lines], dtype=float)
        mean = np.array([line.split()[3] for line in lines], dtype=float)
        rms = np.array([line.split()[4] for line in lines], dtype=float)

    plt.plot(eta, mean, label=r"$<\phi>$", color='red')
    plt.plot(eta, rms, label=r"$\sqrt{<\phi^2>}$", color='blue')
    plt.xlabel(r"$\eta$")
    plt.legend()
    # plt.ylabel()
    # plt.title(r"Dependence of $<\phi>$ on $\eta$")
    
    plt.savefig(f"{folder_name}/meanphi_rms_eta.png")
    print("mean field and rms png file saved.")
    plt.show() if show_plots else None 
    plt.clf()
else:
    pass

if plot_mean_pmsigma == True:
    with open(f"{folder_name}/{filename_data}") as fil_mean:
        lines = fil_mean.readlines()
        eta = np.array([line.split()[0] for line in lines], dtype=float)
        mean = np.array([line.split()[3] for line in lines], dtype=float)
        sigma = np.array([line.split()[-2] for line in lines], dtype=float)

    plt.plot(eta, mean, color='red')
    plt.fill_between(eta, mean-sigma, mean+sigma, color='red', alpha=0.2)
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$\langle \phi \rangle \pm \sigma_{\phi}$")
    # plt.title(r"Dependence of $<\phi>$ on $\eta$")
    
    plt.savefig(f"{folder_name}/meanphi_pmsigma_eta.png")
    print("mean field and pmsigma png file saved.")
    plt.show() if show_plots else None 
    plt.clf()
else:
    pass

if plot_DWlength == True:
    with open(f"{folder_name}/{filename_data}") as fil_dw:
        lines = fil_dw.readlines()
        eta = np.array([line.split()[0] for line in lines], dtype=float)
        dwlL2 = np.array([line.split()[-1] for line in lines], dtype=float)

    plt.loglog(eta, dwlL2, label=r"$l_{\rm dw}/L^2$", color='green')
    plt.loglog(eta, 1.5/eta, label=r"$\eta^{-1}$", linestyle='--', color='black')
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$l_{\rm dw}/L^2$")
    plt.title(r"Dependence of $l_{\rm dw}/L^2$ on $\eta$")
    plt.legend()
    plt.savefig(f"{folder_name}/DWlength_eta.png")
    print("DWlength eta png file saved.")
    plt.show() if show_plots else None
    plt.clf()
else:
    pass
