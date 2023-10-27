using IntervalArithmetic
using JuMP
import HiGHS

include("./crisp-pcm.jl")
include("./nearly-equal.jl")
include("./importance-estimation.jl")

# 区間重要度の中心を

LPResult_Individual = @NamedTuple{
    # 区間重みベクトル
    wᴸ_barbar::Vector{T}, wᵁ_barbar::Vector{T},
    W::Vector{Interval{T}}, # ([wᵢᴸ, wᵢᵁ])
} where {T <: Real}

# 任意の行と列を削除
function remove_row_col(A::Matrix{T}, row::Int, col::Int)::Matrix{T} where {T <: Real}
    m, n = size(A)

    # 行を除外
    new_matrix = A[setdiff(1:m, row), :]
    # 列を除外
    result_matrix = new_matrix[:, setdiff(1:n, col)]
        
    return result_matrix
end

function phase1_test(A::Matrix{T}, method::Function)::Matrix{T} where {T <: Real}

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    m, n = size(A)

    # Phase 1
    Wᶜ = Matrix{T}(undef, m, n) 
    for k = 1:n
        removed_matrix = remove_row_col(A, k, k)
        solution = method(removed_matrix)
        W = solution

        # Wᶜₖを1として挿入
        insert!(W, k, 1.0)
        Wᶜ[:, k] = W
    end

    return Wᶜ
end

# 提案手法
# methodにはAD, EV, GMを指定
function solveIntervalAHP(A::Matrix{T}, method::Function)::LPResult_Individual{T} where {T <: Real}
    ε = 1e-8 # << 1

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    m, n = size(A)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    wᴸ = Matrix{T}(undef, m, n) 
    wᵁ = Matrix{T}(undef, m, n) 
    try
        # Phase 1
        Wᶜ = Matrix{T}(undef, m, n) 
        for k = 1:n
            removed_matrix = remove_row_col(A, k, k)
            solution = method(removed_matrix)
            W = solution

            # Wᶜₖを1として挿入
            insert!(W, k, 1.0)
            Wᶜ[:, k] = W
        end

        # Phase 2
        d_star = T[]
        for k = 1:n
            @variable(model, l[i=1:n] ≥ ε)
            @variable(model, μₖ)
            # 配列Mをμₖで定義し、k番目だけ1-μₖに設定
            μ = fill(μₖ, n)
            μ[k] = 1-μₖ
            
            d_kbar = 0
            for j = 1:n
                sum_plus = 0
                sum_minus = 0
                for i = filter(i -> j != i, 1:n)
                    aᵢⱼ = A[i,j]
                    @constraint(model, aᵢⱼ*(μ[j]*Wᶜ[j,k]-l[j]) ≤ μ[i]*Wᶜ[i,k]+l[i])
                    sum_plus += μ[i]*Wᶜ[i,k] + l[i]
                    sum_minus += μ[i]*Wᶜ[i,k] - l[i]
                end
                @constraint(model, sum_plus + μ[j]*Wᶜ[j,k] - l[j] ≥ 1)
                @constraint(model, sum_minus + μ[j]*Wᶜ[j,k] + l[j] ≤ 1)
                @constraint(model, μ[j]*Wᶜ[j,k] - l[j] ≥ 0)
                
                if j != k
                    d_kbar = l[j]
                end
            end
            @objective(model, Min, d_kbar)
            optimize!(model)
            d_star[k] = value.d_kbar
        end

        # Phase 3
        μ_star = T[]
        l_star = Matrix{T}(undef, m, n) 
        for k = 1:n
            @variable(model, l[i=1:n] ≥ ε)
            @variable(model, μₖ)
            # 配列Mをμₖで定義し、k番目だけ1-μₖに設定
            μ = fill(μₖ, n)
            μ[k] = 1-μₖ
            
            d_kbar = 0
            for j = 1:n
                sum_plus = 0
                sum_minus = 0
                for i = filter(i -> j != i, 1:n)
                    aᵢⱼ = A[i,j]
                    @constraint(model, aᵢⱼ*(μ[j]*Wᶜ[j,k]-l[j]) ≤ μ[i]*Wᶜ[i,k]+l[i])
                    sum_plus += μ[i]*Wᶜ[i,k] + l[i]
                    sum_minus += μ[i]*Wᶜ[i,k] - l[i]
                end
                @constraint(model, sum_plus + μ[j]*Wᶜ[j,k] - l[j] ≥ 1)
                @constraint(model, sum_minus + μ[j]*Wᶜ[j,k] + l[j] ≤ 1)
                @constraint(model, μ[j]*Wᶜ[j,k] - l[j] ≥ 0)
                
                if j != k
                    d_kbar = l[j]
                end
            end
            @constraint(model, d_kbar = d_star[k])
            @objective(model, Min, l[k])
            optimize!(model)
            μ_star = value.μₖ
            l_star = value.l'


            for i = 1:n
                wᴸ[i, k] = μ_star*Wᶜ[i,k] - l_star[i]
                wᵁ[i, k] = μ_star*Wᶜ[i,k] + l_star[i]
            end
        end

        # 最適値
        for i = 1:n
            wᴸ_barbar[i] = min(wᴸ[i, :])
            wᵁ_barbar[i] = max(Wᵁ[i, :])
        end
        W_value = map(i -> (wᴸ_barbar[i])..(wᵁ_barbar[i]), 1:n)

        return (
            wᴸ=wᴸ_barbar, wᵁ=wᵁ_barbar,
            W=W_value
        )
    finally
        # エラー終了時にも変数などを消去する
        empty!(model)
    end
end