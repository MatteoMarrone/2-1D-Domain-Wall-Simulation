This program solves the equation for a scalar field that comes from a double well potential, over a cosmological background, using the technique of fattening to change the dynamical equation of motion.

The spatial derivatives are calculated with a finite difference scheme, and the time evolution is done by a RK4 solver.

Initially one need to specify the relevant parameters for the simulation: cosmological parameters and also numerical parameters, like the lattice size and resolution. Then what the program does is evolve in time the initial
conditions for the field "phi" and the first derivative "dtphi", up until the final time of simulation. During the simulation, every "save_step" steps, the field configuration is saved in .npy file. 

At the end of the simulation we read the .npy files and do an animation of the time evolution of the system. We also collect the data stored during the simulation regarding the average value of the field, rms, etc., and plot
them over time to see the evolution of relevant quantities. Finally, the mp4 files of the animation as well as the png files of the plots are saved. 

One can also use the program without having to do the animations, for example, by simply setting "anim_phi = False" initially. By doing this, the program will omit the process of creating and saving the animation of the
time evolution of the field. Also, if one does not want to plot the data, set "plot_... = False", etcetera.
