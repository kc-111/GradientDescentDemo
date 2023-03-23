using DifferentialEquations
using Plots
using Zygote
using OptimizationDemo
using ProgressMeter
using SciMLSensitivity

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

# Now let's find the correct costates λ_i
anim = Animation() # Initialize animation
tspan = (0, 4)
itrs = 1000
λ_i = -rand(3)
my_opt = my_AdaBelief(λ_i, 0.05; β=(0.9, 0.999))
pbar = ProgressUnknown(desc="Optimizing...", spinner=true)
lossvals = []
for i=1:itrs
    global anim, u0, λ_i, my_p
    _loss = 0.
    _grads = Zygote.gradient(λ_i) do x
        Zygote.forwarddiff(x) do inp
            _prob = ODEProblem(the_orbit, vcat(u0, inp), tspan, my_p)
            _sol = solve(_prob, AutoTsit5(Rosenbrock23(autodiff=false)))
            _t = _sol.t
            _sol = mapreduce(permutedims, vcat, _sol.u)
            Zygote.ignore_derivatives() do 
                # Need to convert dual numbers to real values to plot it
                _plotsol = zeros(size(_sol))
                _plott = zeros(size(_t))
                for i=1:size(_sol)[1]
                    _plott[i] = _t[i]
                    for j=1:size(_sol)[2]
                        _plotsol[i, j] = _sol[i, j].value
                    end
                end
                curframe = plot(_plott, _plotsol[:, 1:3], 
                    labels = ["r" "u" "v"],
                    color = [:red :blue :green],
                    ylims=(-0.5, 1.7))
                scatter!([_plott[end]], [0. sqrt(my_p[1]/_plotsol[end, 1])], 
                    labels = ["u_f" "v_f"],
                    color = [:blue :green],
                    legend=:outertopright)
                frame(anim)
            end
            _loss = my_loss(_sol[end, :], my_p)
            return _loss
        end
    end
    λ_i .-= my_opt(_grads[1], λ_i) # Update
    next!(pbar; showvalues = [
        (:Iteration, string(i, "/", itrs)), 
        (:Loss, string(_loss.value))
        ]
    )
    push!(lossvals, _loss.value)
end
finish!(pbar)
display(lossvals[end])
display(plot(lossvals))
# Display animations
gif(anim, "shooting.gif", fps = 30)