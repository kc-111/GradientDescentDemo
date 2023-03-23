using DifferentialEquations
using Plots
using Zygote
using OptimizationDemo
using ProgressMeter
using SciMLSensitivity
using Flux

function the_orbit(u, p, t)
    mu = p[1]
    m0 = p[2]
    m1 = p[3]
    T = p[4]
    r = u[1]
    u_var = u[2]
    v = u[3]
    λr = u[4]
    λu = u[5]
    λv = u[6]

    sinϕ = -λu/sqrt(λu^2 + λv^2)
    cosϕ = -λv/sqrt(λu^2 + λv^2)

    drdt = u_var
    dudt = v^2/r - mu/r^2 + T*sinϕ/(m0 + m1*t)
    dvdt = -u_var*v/r + T*cosϕ/(m0 + m1*t)
    dλrdt = -λu*(-v^2/r^2 + 2*mu/r^3) - λv*(u_var*v/r^2);
    dλudt = -λr + λv*v/r;
    dλvdt = -λu*2*v/r + λv*u_var/r;

    dλrdt = -λu*(-v^2/r^2 + 2*mu/r^3) - λv*(u_var*v/r^2);
    return vcat(drdt, dudt, dvdt, dλrdt, dλudt, dλvdt)
end

# Initial bounds
# r_i = 1
# u_i = 0
# v_i = sqrt(mu/r_i);
# Bounds at the final time point
# u_f = 0;
# v_f = sqrt(mu/r_f);
# λr_f + 1 - λv_f*sqrt(mu)/2/r_f^(3/2) = 0

# Parameters
mu = 1 
m0 = 1
m1 = -0.1497
T = 0.07025
my_p = [mu, m0, m1, T]

# Initial states
r_i = 1
u_i = 0
v_i = 1

# Initial Guess for the noncostates
u0 = [r_i, u_i, v_i]

# Loss function
function my_loss(uf, p)
    cost1 = uf[2]
    cost2 = uf[3] - sqrt(p[1]/uf[1])
    cost3 = uf[4] + 1 - uf[6]*sqrt(p[1])/2/uf[1]^(3/2)
    return abs2(cost1) + abs2(cost2) + abs2(cost3) # Using abs causes instabilities
end

# Multiple shooting where tspan is our grid points
function multi_shoot(initial_u0, u_guess, λ_guess, tspan, my_p)
    t_int = tspan[2] # Assume ODE not dependent on time
    multi_sols = Matrix{Real}[] # Store the "continuous" solutions
    multi_sols_finals = Vector{Real}[] # Store the solutions at final points
    multi_sols_t = Vector{Real}[] # Store the time points
    cur_t = 0.
    for i=1:(size(tspan)[1] - 1) # If there are 10 time points, we integrate 9 times
        if i == 1 # In the initial integration, we use initial_u0 and λ_guess[1, :]
            _prob = ODEProblem(the_orbit, vcat(initial_u0, λ_guess[i, :]), (cur_t, cur_t+t_int), my_p)
        else # We have to use the guess at this point
            _prob = ODEProblem(the_orbit, vcat(u_guess[i-1, :], λ_guess[i, :]), (cur_t, cur_t+t_int), my_p)
        end
        _sol = solve(_prob, AutoTsit5(Rosenbrock23(autodiff=false)))
        _t = _sol.t
        _sol = mapreduce(permutedims, vcat, _sol.u)
        push!(multi_sols_t, _t)
        push!(multi_sols, _sol)
        push!(multi_sols_finals, _sol[end, :])
        cur_t += t_int
    end
    multi_sols_finals = mapreduce(permutedims, vcat, multi_sols_finals)
    return multi_sols_finals, multi_sols_t, multi_sols
end

function multi_shoot_loss(initial_u0, u_guess, λ_guess, tspan, my_p, penalty)
    multi_sols_finals, multi_sols_t, multi_sols = multi_shoot(initial_u0, u_guess, λ_guess, tspan, my_p)
    # First is the final time loss
    loss_orig = my_loss(multi_sols_finals[end, :], my_p)
    penalized_loss = loss_orig
    # Second is the continuity loss
    for i=1:size(multi_sols_finals)[1]-1
        penalized_loss += penalty*sum(abs2, multi_sols_finals[i, :] - vcat(u_guess[i, :], λ_guess[i+1, :]))
    end
    return loss_orig, penalized_loss, multi_sols_t, multi_sols
end

# Set up
points = 3
tspan = collect(range(0, 6, points))
multi_u_guess = 1 .+ rand(points-2, 3)
multi_λ_guess = -0.1*rand(points-1, 3)
my_u0 = vcat(u0, multi_λ_guess[1, :])
multi_u_guess = collect(Iterators.flatten(multi_u_guess))
multi_λ_guess = collect(Iterators.flatten(multi_λ_guess))
multi_u_size = length(multi_u_guess)
combined_guesses = vcat(multi_u_guess, multi_λ_guess)

# Now let's find the correct costates λ_i
anim = Animation() # Initialize animation
itrs = 1000
my_opt = my_AdaBelief(combined_guesses, 0.1; β=(0.9, 0.999))
cont_penalty = 10 .* sigmoid_fast.(2. * LinRange(-5, 5, itrs)) # Too heavy of a penalty may not work!
pbar = ProgressUnknown(desc="Optimizing...", spinner=true)
lossvals = []
for i=1:itrs
    global anim, u0, combined_guesses, multi_u_size, my_p, tspan, cont_penalty
    _loss = 0.
    _plott = []
    _plotsols = []
    _grads = Zygote.gradient(combined_guesses) do x
        Zygote.forwarddiff(x) do inp
            inp1 = reshape(inp[1:multi_u_size], points-2, 3)
            inp2 = reshape(inp[multi_u_size+1:end], points-1, 3)
            _loss, _useloss, _plott, _plotsols = multi_shoot_loss(u0, inp1, inp2, tspan, my_p, cont_penalty[i])
            return _useloss
        end
    end
    combined_guesses .-= my_opt(_grads[1], combined_guesses) # Update

    # Plot
    _plotsol_use = Matrix{Float64}[]
    for _sols in _plotsols
        _solsmat = zeros(size(_sols))
        for i=1:size(_sols)[1]
            for j=1:size(_sols)[2]
                _solsmat[i, j] = _sols[i, j].value
            end
        end
        push!(_plotsol_use, _solsmat)
    end

    _p = plot()
    for plotvals in zip(_plott, _plotsol_use)
        plot!(plotvals[1], plotvals[2][:, 1:3], 
            labels = ["r" "u" "v"],
            color = [:red :blue :green],
            legend = false,
            ylims=(-0.5, 2.5))
    end
    scatter!([_plott[end][end]], [0. sqrt(my_p[1]/_plotsol_use[end][end, 1])], 
        labels = ["u_f" "v_f"],
        legend = false,
        color = [:blue :green])
    frame(anim)

    # Save loss
    push!(lossvals, _loss.value)

    # Go next
    next!(pbar; showvalues = [
        (:Iteration, string(i, "/", itrs)), 
        (:Loss, string(_loss.value))
        ]
    )
end
finish!(pbar)
display(lossvals[end])
display(plot(lossvals))
# Display animations
gif(anim, string("multishooting_n", points, "_t", tspan[end] ,".gif"), fps = 30)