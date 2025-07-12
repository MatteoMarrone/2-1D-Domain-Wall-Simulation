This program solves the equation of motion for a 2D scalar field arising from a double-well potential, evolving over a cosmological background. The dynamical equation is modified using the fattening technique to enhance the resolution of domain walls.

Spatial derivatives are computed using a finite-difference scheme, and time evolution is performed using a 4th-order Runge–Kutta (RK4) integrator.

Before starting the simulation, the user must specify both cosmological parameters (e.g., expansion rate, scale factor behavior...), and numerical parameters (e.g., lattice size, grid resolution, time step...). Given these, the program evolves the scalar field phi, and its first time derivative, from the initial conditions, up to a specified final time. During the evolution, every "save_step" steps, the field configuration is saved in .npy format.

After the simulation, the saved .npy files are used to create an animation of the time evolution of the field. Additional data recorded during the simulation (e.g., mean value of the field, RMS fluctuations) are read and plotted over time to track the evolution of relevant physical quantities. Both the animation (as an .mp4 file) and the plots (as .png files) are saved to disk.

The user can control which outputs are generated: To skip the animation, set anim_phi = False at the beginning of the program (as well as the other possible animations anim_V, anim_gradsqphi ...). To skip plotting any specific quantity, set the corresponding plot_... = False. And so on.
