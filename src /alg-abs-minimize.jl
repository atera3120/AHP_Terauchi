using IntervalArithmetic
using JuMP
import HiGHS

include("./crisp-pcm.jl")
include("./nearly-equal.jl")

LPResult_Individual = @NamedTuple{
    # 区間重みベクトル
    wᴸ::Vector{T}, wᵁ::Vector{T},
    W::Vector{Interval{T}}, # ([wᵢᴸ, wᵢᵁ])
    optimalValue::T
    } where {T <: Real}

function AbsErrMin(A::Matrix{T})::T where {T <: Real}
    ε = 1e-8 # << 1

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    m, n = size(A)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        @variable(model, u[i=1:n]);
        @variable(model, U[i=1:n; j=i+1:n]);
        @variable(model, ∑∑Uᵢⱼ);
        
        # 上三角成分に対応する i, j
        ∑∑Uᵢⱼ = 0
        for i = 1:n-1
            for j = i+1:n
                wᵢᴸ = wᴸ[i]; wᵢᵁ = wᵁ[i]
                uᵢ = u[i]; uⱼ = u[j]; Uᵢⱼ = U[i,j]
                aᵢⱼ = A[i,j]
                @constraint(model, log(e, aᵢⱼ) - uᵢ + uⱼ <= Uᵢⱼ)
                @constraint(model, -log(e, aᵢⱼ) + uᵢ - uⱼ <= Uᵢⱼ)
                ∑∑Uᵢⱼ += Uᵢⱼ
            end
        end
        
        # 目的関数 ∑∑Uᵢⱼ
        @objective(model, Min, ∑∑Uᵢⱼ)

        optimize!(model)
        temp = exp(value.(u))
        OptimalValue = sum(temp)
    
    finally
        # エラー終了時にも変数などを消去する
        empty!(model)    
    end
end