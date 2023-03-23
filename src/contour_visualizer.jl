"""
    Given several trajectories, we visualize the descent path of the optimization problem.
    This function returns an animation object that animates the optimization path.
    Note that plotting is slow...

# Arguments
- `surface::Tuple`: The surface of the optimization problem (x, y, z)
- `trajectories::AbstractArray`: The path of the descent with dimensions corresponding to (# time points, # of Optimizers, 3 dimensions)
- `names::AbstractArray`: The names of the optimizers
- `constraints::AbstractArray`: The constraints area or lines with dimensions corresponding to (# points, # of constraints, 3 dimensions)
"""
function visualize_contour(surface::Tuple, trajectories::AbstractArray, names::AbstractArray; 
    constraints::AbstractArray=[])
    colors = cgrad(:Set1_5, categorical=true) # Colors
    constraint_names = String[] # Set some names
    for i=1:size(constraints)[1]
        push!(constraint_names, string("Constraint ", i))
    end
    anim = Animation() # Initialize animation
    gr(size = (800, 500))

    pbar = ProgressUnknown(desc="Creating Animation... ", spinner=true)
    for i=1:size(trajectories)[1]

        # Every point we plot contours and constraints
        curframe = plot(surface[1], surface[2], surface[3], st = :contour, levels=25, fill=true, size = (800, 500))
        for j=1:size(constraints)[1]
            curframe = scatter!(constraints[j][:, 1], constraints[j][:, 2], markerstrokewidth=0, 
                markercolor = :white, markersize=0.5, label="")
        end

        # Plot trajectory
        for j=1:size(trajectories)[2] # For every optimizer
            curframe = plot!(
                trajectories[1:i, j, 1], 
                trajectories[1:i, j, 2], 
                labels = names[j], 
                linecolor = colors[j],
                linewidth = 2, 
                linealpha = 0.75,
                legend = :outertopleft
            )
            curframe = scatter!(
                [trajectories[i, j, 1]], 
                [trajectories[i, j, 2]], 
                label = "", 
                markercolor = colors[j], 
                markershape = :circle, 
                markerstrokewidth = 2, 
                legend = :outertopleft
            )
        end
        frame(anim)
        next!(pbar; showvalues = [(:Frame, string(i, "/", size(trajectories)[1]))])
    end
    finish!(pbar)
    return anim
end