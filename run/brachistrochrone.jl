using Revise
using OptimizationDemo
using Zygote
using ProgressMeter
using Plots
using Interpolations

"""
    Cost function for solving the brachistrochrone problem. Points need to be in the fourth quadrant.

`p`: Parameters or points that are equally spaced on a grid defined by the user (y points)
`spacing`: The spacing between x points
`yi`: Initial position of y
`yf`: Final position of y

"""
function cost_fun(p, spacing, yi, yf)
    # NOTE: Use . in front of the operations where elementwise is needed
    fully_arr = abs.(vcat(yi, p, yf))
    # Compute velocity at each point using the relationship g(hf - hi) = -1/2(vf^2 - vi^2) where hi is 0 and vi is 0
    # Hence, -gy = 1/2v^2 and v = ± sqrt(-2gy) => v = sqrt(-2gy).
    vs = sqrt.(2*9.80665 .* fully_arr) # Velocities
    avg_v = (vs[2:end] .+ vs[1:end-1]) ./ 2 # Average velocities using avg_v = (vi+vf)/2 that is ONLY TRUE IF ACCELERATION IS CONSTANT.
    # To compute time, we use the formula where vt = d where v is average velocity, t is time, and d is distance
    # So t = d/v and d = sqrt(x^2 + y^2) = sqrt(spacing^2 + Δy^2)
    d = sqrt.(abs2(spacing) .+ abs2.(fully_arr[2:end] .- fully_arr[1:end-1]))
    return sum(d ./ avg_v) # Numerically unstable if avg_v -> 0, but whatever
end

function cycloid(t) # Starting from 0, ending at 0 from x=0 to x=150. t ~ 6.25
    return [0.5*6.91^2*(t-sin(t)), 0.5*6.91^2*(1-cos(t))]
end

# Set up
t = collect(range(0, 6.25, 100))
actual_sol = cycloid.(t)
actual_sol = mapreduce(permutedims, vcat, actual_sol)
yi = 0.
yf = 0.
xf = 150.
pts = 50
grid_pts = collect(range(0, xf, pts))
spacing = grid_pts[2]
interp_linear = linear_interpolation(actual_sol[:, 1], actual_sol[:, 2]) # Interpolate actual solution
# params_y = interp_linear(grid_pts[2:end-1]) # Start at true solution
params_y = 50 .* rand(length(grid_pts)-2) .+ 0.5 # Exclude initial and final ys
params_y2 = deepcopy(params_y)
my_opt = my_AdaBelief(params_y, 5.0; β=(0.9, 0.999))
my_opt2 = my_sgdm(params_y, 5.0; β=0.)
numiters = 1000
@show size(params_y)

# Train
anim = Animation()
pbar = ProgressUnknown(desc="Optimizing...", spinner=true)
for i=1:numiters
    global anim, params_y, params_y2, my_opt, my_opt2, spacing, yi, yf, xf, actual_sol
    grads = Zygote.gradient(params_y) do p # Forward differentiation will nan due to sqrt(abs())
        return cost_fun(p, spacing, yi, yf)
    end
    params_y .-= my_opt(grads[1], params_y) # Update
    grads = Zygote.gradient(params_y2) do p # Forward differentiation will nan due to sqrt(abs())
        return cost_fun(p, spacing, yi, yf)
    end
    params_y2 .-= my_opt2(grads[1], params_y2) # Update

    # Plot
    curframe = scatter([grid_pts[1]], [yi], markershape = :circle, markerstrokewidth = 2, label="Initial Point")
    curframe = scatter!([grid_pts[end]], [yf], markershape = :circle, markerstrokewidth = 2, label="Final Point")
    curframe = plot!(
        actual_sol[:, 1], 
        -actual_sol[:, 2], 
        label="Actual Solution", 
        linewidth = 2, 
        linestyle = :dash,
        grid=false
    )
    curframe = plot!(
        grid_pts, 
        -abs.(vcat(yi, params_y, yf)), 
        label="Predicted Solution with Momentum", 
        linewidth = 2, 
        grid=false
    )
    curframe = plot!(
        grid_pts, 
        -abs.(vcat(yi, params_y2, yf)), 
        label="Predicted Solution without Momentum", 
        linewidth = 2, 
        grid=false
    )
    frame(anim)
    next!(pbar; showvalues = [(:Iteration, string(i, "/", numiters)), ])
end
finish!(pbar)
gif(anim, "brachistrochrone_trajectory.gif", fps = 30)

# Compute t costs. True cost is not exact.
cost = cost_fun(params_y, spacing, yi, yf)
true_cost = cost_fun(interp_linear(grid_pts[2:end-1]), spacing, yi, yf)
println(string("Cost(t): ", cost))
println(string("Percentage Error: ", round((cost - true_cost)/true_cost*100, digits=2), "%"))