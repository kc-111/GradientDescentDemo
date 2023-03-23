using Revise
using OptimizationDemo
using Zygote
using ProgressMeter
using Plots

# start = [0.; 0.1] # Himmelblau
# start = [2.5; 5.0] # Saddle flat
# start = [-2.0; -2.5] # Himmelblau
# start = [-2.5; -5.5] # Slanted ellipse
# start = [0.1; 5.] # Rosenbrock
# start = [2.4; 5.] # Rosenbrock 2
# start = [2.2; 5.] # Rosenbrock 3
start = [-2.5; 0.0] # Rosenbrock 4

# Create objective surface
# surface_ellipse, surface_rosenbrock (log), surface_himmelblau (log), surface_saddleflat
use_log = true
my_obj = surface_rosenbrock 
graphing_pts = 1000
grid_points = range(start=-6, stop=6, length=graphing_pts)
if use_log # Sometimes, the surface needs to be scaled
    my_surface = log.(my_obj.(grid_points', grid_points))
else
    my_surface = my_obj.(grid_points', grid_points)
end
my_surface = (grid_points, grid_points, my_surface)

# Initialize optimizer
numiters = 200
my_opts = []
push!(my_opts, my_sgdm(start, 0.01; β=0.9))
push!(my_opts, my_AdaBelief(start, 0.5; β=(0.95, 0.999)))
push!(my_opts, my_Adam(start, 0.5; β=(0.95, 0.999)))
push!(my_opts, my_Adamax(start, 1.0; β=(0.95, 0.999)))
push!(my_opts, my_AdaBeliefmax(start, 1.0; β=(0.95, 0.999)))
θs = zeros(length(my_opts), 2)
for i=1:length(my_opts)
    θs[i, :] = start
end
trajectory_opts = zeros(numiters, length(my_opts), 3)
pbar = ProgressUnknown(desc="Optimizing...", spinner=true)

# Start optimizing
for i=1:numiters
    global θs
    global trajectory_opts

    for j=1:size(my_opts)[1]
        gs = gradient(θs[j, :]) do θ # Compute gradient
            _loss = my_obj(θ...)
            Zygote.ignore_derivatives() do
                trajectory_opts[i, j, :] = [θ[1], θ[2], _loss]
            end
            return _loss
        end
        θs[j, :] .-= my_opts[j](gs[1], θs[j, :]) # Update
    end
    next!(pbar; showvalues = [(:Iteration, string(i, "/", numiters)), ])
end
finish!(pbar)

# Plot animation
names = []
for i=1:length(my_opts)
    push!(names, string("Optimizer ", i))
end
anim = visualize_contour(my_surface, trajectory_opts, names)
# Display animations
gif(anim, "opt_traj_unconstr.gif", fps = 15)