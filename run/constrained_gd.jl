using Revise
using OptimizationDemo
using Zygote
using ProgressMeter
using Plots

# start = [0.; 0.1] # Himmelblau
# start = [-2.5; 1.] # Saddle flat
# start = [-2.0; -2.5] # Himmelblau
# start = [-2.5; -5.5] # Slanted ellipse
# start = [0.1; 5.] # Rosenbrock
start = [2.4; 5.] # Rosenbrock 2
# start = [2.2; 5.] # Rosenbrock 3

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

# Create constraints
my_constraint_fs = []
push!(my_constraint_fs, (x, y) -> 20*(x-2.5)^2*(y-2.5)^2/((x-2.5)^2+(y-2.5)^2)^3 - 1) # 20*x^2*y^2/(x^2+y^2)^3 = 1, h(x) = 20*x^2*y^2/(x^2+y^2)^3 - 1
push!(my_constraint_fs, (x, y) -> y + x - 2.5) # y = - (x-2.5), h(x) = - y - x + 2.5
my_constraint = Matrix{Float64}[]
for i=1:size(my_constraint_fs)[1] # Number of constraints
    lineseg = Vector{Float64}[]
    for j=1:size(grid_points)[1]
        for k=1:size(grid_points)[1]
            if abs(round(my_constraint_fs[i](grid_points[j], grid_points[k]); digits=1)) == 0.
                push!(lineseg, [grid_points[j], grid_points[k], my_obj(grid_points[j], grid_points[k])])
            end
        end
    end
    push!(my_constraint, mapreduce(permutedims, vcat, lineseg))
end
constraint_types = [1, 2] # 1: ≥, 2: =, 3: ≤

# Initialize optimizer
numiters = 200
my_opts = []
push!(my_opts, my_sgdm(start, 0.1; β=0.0))
push!(my_opts, my_sgdm(start, 0.1; β=0.9))
push!(my_opts, my_AdaBelief(start, 0.1; β=(0.9, 0.999)))
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
            _loss = surface_ellipse(θ...)
            Zygote.ignore_derivatives() do
                trajectory_opts[i, j, :] = [θ[1], θ[2], _loss]
            end
            weight = 1
            for k=1:size(my_constraint_fs)[1] # Weighted loss
                # Inequality or equality constraints
                if constraint_types[k] == 1
                    constr_loss = my_constraint_fs[k](θ...)
                    if constr_loss >= 0.
                        constr_loss = 0.
                    else
                        constr_loss = 5*abs(constr_loss)
                        weight = weight + 5
                    end
                elseif constraint_types[k] == 2
                    constr_loss = 5*abs(my_constraint_fs[k](θ...))
                    weight = weight + 5
                else
                    constr_loss = my_constraint_fs[k](θ...)
                    if constr_loss <= 0.
                        constr_loss = 0.
                    else
                        constr_loss = 5*abs(constr_loss)
                        weight = weight + 5
                    end
                end
                _loss = _loss + constr_loss
            end
            return _loss/weight
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
anim = visualize_contour(my_surface, trajectory_opts, names; constraints=my_constraint)
# Display animations
gif(anim, "optimiser_trajectory.gif", fps = 15)