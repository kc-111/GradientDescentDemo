"""
    Adabelief optimiser implemented by me.

# Arguments
- `θ::AbstractArray`: Parameters
- `α::Float64`: Learning rate
- `β::Tuple{Float64, Float64}`: Smoothing parameters
- `λ1::Float64`: Weight decay L1
- `λ2::Float64`: Weight decay L2
- `ϵ::Float64`: Denominator to prevent division by 0
"""
mutable struct my_AdaBelief <: Function
    α::Float64 # learning rate
    β::Tuple{Float64, Float64} # Smoothing parameters
    ϵ::Float64 # Needed to prevent explosion
    m::AbstractArray # Exponential moving average of gradient
    s::AbstractArray # Exponential moving average of (gradient - momentum)^2
    λ1::Float64 # Decoupled weight decay L1
    λ2::Float64 # Decoupled weight decay L2
    t::Float64 # Current iteration
end

my_AdaBelief(
    θ::AbstractArray,
    α::Float64;
    β::Tuple{Float64, Float64} = (0.95, 0.999),
    λ1::Float64 = 0.0,
    λ2::Float64 = 0.0,
    ϵ::Float64 = 1e-16
) = my_AdaBelief(α, β, ϵ, zero(θ), zero(θ), λ1, λ2, 0.0)

function (optstate::my_AdaBelief)(Δ::AbstractArray, θ::AbstractArray) 
    # Δ is gradient, θ is parameters, η is schedule multiplier for learning rate and momentum beta1.
    optstate.t += 1
    @. optstate.m = optstate.β[1]*optstate.m + (1 - optstate.β[1])*Δ # Mean
    @. optstate.s = optstate.β[2]*optstate.s + (1 - optstate.β[2])*(Δ - optstate.m)^2 # Centered variance
    # Bias correction
    m_corr = @. optstate.m/(1 - optstate.β[1]^optstate.t)
    s_corr = @. optstate.s/(1 - optstate.β[2]^optstate.t)
    descent = @. optstate.α*(m_corr/(√s_corr + optstate.ϵ) + optstate.λ1*sign(θ) + optstate.λ2*θ) # Elastic Net

    return descent
end

"""
    Adam optimiser implemented by me.

# Arguments
- `θ::AbstractArray`: Parameters
- `α::Float64`: Learning rate
- `β::Tuple{Float64, Float64}`: Smoothing parameters
- `λ1::Float64`: Weight decay L1
- `λ2::Float64`: Weight decay L2
- `ϵ::Float64`: Denominator to prevent division by 0
"""
mutable struct my_Adam <: Function
    α::Float64 # learning rate
    β::Tuple{Float64, Float64} # Smoothing parameters
    ϵ::Float64 # Needed to prevent explosion
    m::AbstractArray # Exponential moving average of gradient
    s::AbstractArray # Exponential moving average of (gradient - momentum)^2
    λ1::Float64 # Decoupled weight decay L1
    λ2::Float64 # Decoupled weight decay L2
    t::Float64 # Current iteration
end

my_Adam(
    θ::AbstractArray,
    α::Float64;
    β::Tuple{Float64, Float64} = (0.95, 0.999),
    λ1::Float64 = 0.0,
    λ2::Float64 = 0.0,
    ϵ::Float64 = 1e-16
) = my_Adam(α, β, ϵ, zero(θ), zero(θ), λ1, λ2, 0.0)

function (optstate::my_Adam)(Δ::AbstractArray, θ::AbstractArray) 
    # Δ is gradient, θ is parameters, η is schedule multiplier for learning rate and momentum beta1.
    optstate.t += 1
    @. optstate.m = optstate.β[1]*optstate.m + (1 - optstate.β[1])*Δ # Mean
    @. optstate.s = optstate.β[2]*optstate.s + (1 - optstate.β[2])*Δ^2 # Raw variance
    # Bias correction
    m_corr = @. optstate.m/(1 - optstate.β[1]^optstate.t)
    s_corr = @. optstate.s/(1 - optstate.β[2]^optstate.t)
    descent = @. optstate.α*(m_corr/(√s_corr + optstate.ϵ) + optstate.λ1*sign(θ) + optstate.λ2*θ) # Elastic Net

    return descent
end

"""
    Adamax optimiser implemented by me.

# Arguments
- `θ::AbstractArray`: Parameters
- `α::Float64`: Learning rate
- `β::Tuple{Float64, Float64}`: Smoothing parameters
- `λ1::Float64`: Weight decay L1
- `λ2::Float64`: Weight decay L2
- `ϵ::Float64`: Denominator to prevent division by 0
"""
mutable struct my_Adamax <: Function
    α::Float64 # learning rate
    β::Tuple{Float64, Float64} # Smoothing parameters
    ϵ::Float64 # Needed to prevent explosion
    m::AbstractArray # Exponential moving average of gradient
    s::AbstractArray # Exponential moving average of (gradient - momentum)^2
    λ1::Float64 # Decoupled weight decay L1
    λ2::Float64 # Decoupled weight decay L2
    t::Float64 # Current iteration
end

my_Adamax(
    θ::AbstractArray,
    α::Float64;
    β::Tuple{Float64, Float64} = (0.95, 0.999),
    λ1::Float64 = 0.0,
    λ2::Float64 = 0.0,
    ϵ::Float64 = 1e-16
) = my_Adamax(α, β, ϵ, zero(θ), zero(θ), λ1, λ2, 0.0)

function (optstate::my_Adamax)(Δ::AbstractArray, θ::AbstractArray) 
    # Δ is gradient, θ is parameters, η is schedule multiplier for learning rate and momentum beta1.
    optstate.t += 1
    @. optstate.m = optstate.β[1]*optstate.m + (1 - optstate.β[1])*Δ # Mean
    @. optstate.s = max(optstate.β[2]*optstate.s, abs(Δ)) # Infinity norm
    # Bias correction
    m_corr = @. optstate.m/(1 - optstate.β[1]^optstate.t)
    descent = @. optstate.α*(m_corr/optstate.s + optstate.λ1*sign(θ) + optstate.λ2*θ) # Elastic Net

    return descent
end

"""
    Standard gradient descent with momentum

# Arguments
- `θ::AbstractArray`: Parameters
- `α::Float64`: Learning rate
- `β::Float64`: Momentum parameter
- `λ1::Float64`: Weight decay L1
- `λ2::Float64`: Weight decay L2
"""
mutable struct my_sgdm <: Function
    α::Float64 # learning rate
    β::Float64 # Momentum parameter
    m::AbstractArray # Exponential moving average of gradient
    λ1::Float64 # Decoupled weight decay L1
    λ2::Float64 # Decoupled weight decay L2
end

my_sgdm(
    θ::AbstractArray,
    α::Float64;
    β::Float64 = 0.9,
    λ1::Float64 = 0.0,
    λ2::Float64 = 0.0
) = my_sgdm(α, β, zero(θ), λ1, λ2)

function (optstate::my_sgdm)(Δ::AbstractArray, θ::AbstractArray) 
    # Δ is gradient, θ is parameters, η is schedule multiplier for learning rate and momentum beta1.
    @. optstate.m = optstate.β*optstate.m + (1 - optstate.β)*Δ # Mean
    descent = @. optstate.α*(optstate.m + optstate.λ1*sign(θ) + optstate.λ2*θ) # Decoupled weight decay L1
    return descent
end