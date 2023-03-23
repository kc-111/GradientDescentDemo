# α>0, α must be a multiple of 0.4. β>1
function compute_softplus_shift_and_threshold(α::Float64=4.0, β::Float64=1.0)
    shift = 1/(β*(α/β)^(α/(α+1))) + (α/β)^(1/(α+1))
    threshold = -(α/β)^(1/(α+1)) + shift
    return shift, threshold
end

function my_softplus(x)
    # You can compute threshold and shift this way
    # shift = 1/(β*(α/β)^(α/(α+1))) + (α/β)^(1/(α+1))
    # threshold = -(α/β)^(1/(α+1)) + shift
    # We use α = 2 because it is fast
    if x < 1.0197152163700611 # Threshold
        val = x-9.17743694733054 # Shift
        return 1/(0.00000005*abs2(val)*abs2(val)*abs2(val)*abs2(val))
    end
    return log(1.0+exp(x+20.0)) - 20.0 #x
end

function my_mish(x)
    # f(x) = x < 0 ? a*x/(a+b*x^c)
    # 0.8x faster than ReLU
    # 2x faster than swish
    # 5x faster than softplus
    # 8x faster than mish
    if x < 0.0
        return 10.0*x/(10.0 + abs2(x)*abs2(x))
    end
    return log(1.0+exp(x+20.0)) - 20.0 #x
end

my_α=0.4*20
my_β=0.00000005
my_shift, my_threshold = compute_softplus_shift_and_threshold(my_α, my_β);
@show my_shift
@show my_threshold

using BenchmarkTools
using Flux

data = randn(1000);
@btime my_softplus.(data)
@btime my_mish.(data)
@btime relu.(data)
@btime softplus.(data)
@btime mish.(data)
@btime swish.(data)

using Plots
foox = LinRange(-5, 5, 100)
fooy = my_softplus.(foox)
plot(foox, fooy)
plot!(foox, softplus.(foox))
plot!(foox, my_mish.(foox))
plot!(foox, mish.(foox))
plot!(foox, swish.(foox))
plot!(foox, relu.(foox))