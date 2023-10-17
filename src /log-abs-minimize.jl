using IntervalArithmetic
using JuMP
import HiGHS

include("./crisp-pcm.jl")
include("./nearly-equal.jl")

function LogAbsErrMin(A::Matrix{T})::T where {T <: Real}
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
                @constraint(model, log(ℯ, aᵢⱼ) - uᵢ + uⱼ <= Uᵢⱼ)
                @constraint(model, -log(ℯ, aᵢⱼ) + uᵢ - uⱼ <= Uᵢⱼ)
                ∑∑Uᵢⱼ += Uᵢⱼ
            end
        end
        
        # 目的関数 ∑∑Uᵢⱼ
        @objective(model, Min, ∑∑Uᵢⱼ)

        optimize!(model)
        temp = exp.(value.(u))
        optimalValue = temp/sum(temp)

        return (
            optimalValue=optimalValue
        )
    
    finally
        # エラー終了時にも変数などを消去する
        empty!(model)    
    end
end