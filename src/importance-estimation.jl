using IntervalArithmetic
using LinearAlgebra
using JuMP
import HiGHS

include("./crisp-pcm.jl")
include("./nearly-equal.jl")

function AD(A::Matrix{T})::Array{T} where {T <: Real}
    ε = 1e-8 # << 1

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    m, n = size(A)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        @variable(model, u[i=1:n]);
        @variable(model, U[i=1:n, j=i+1:n]);
        
        # 上三角成分に対応する i, j
        ∑∑Uᵢⱼ = 0
        for i = 1:n-1
            for j = i+1:n
                uᵢ = u[i]; uⱼ = u[j]; Uᵢⱼ = U[i,j]
                aᵢⱼ = A[i,j]
                @constraint(model, log(aᵢⱼ) - uᵢ + uⱼ <= Uᵢⱼ)
                @constraint(model, -log(aᵢⱼ) + uᵢ - uⱼ <= Uᵢⱼ)
                ∑∑Uᵢⱼ += Uᵢⱼ
            end
        end
        
        # 目的関数 ∑∑Uᵢⱼ
        @objective(model, Min, ∑∑Uᵢⱼ)

        optimize!(model)
        exp_conv_array = exp.(value.(u))
        Σexp_conv = sum(exp_conv_array)

        return W = exp_conv_array / Σexp_conv
    
    finally
        # エラー終了時にも変数などを消去する
        empty!(model)    
    end
end

function EV(A::Matrix{T})::Array{T} where {T <: Real}

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    eigen_result = eigen(A)
    λₘₐₓ = maximum(real(eigen_result.values))
    idxₘₐₓ = argmax(real(eigen_result.values))
    vₘₐₓ = eigen_result.vectors[:, idxₘₐₓ]

    return W = vₘₐₓ / sum(vₘₐₓ)

end

function GM(A::Matrix{T})::Array{T} where {T <: Real}
    m, n = size(A)

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    S = 0
    for i = 1:n
        S += prod(A[i, :])^(1/n)
    end

    w = similar(A, T, n)
    for i = 1:n
        w[i] = prod(A[i, :])^(1/n) / S
    end

    return W = w

end