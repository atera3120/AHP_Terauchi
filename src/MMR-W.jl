using IntervalArithmetic
using JuMP
import HiGHS

include("./crisp-pcm.jl")
include("./nearly-equal.jl")
include("./importance-estimation.jl")


LPResult_Individual = @NamedTuple{
    # 区間重みベクトル
    wᴸ::Vector{T}, wᵁ::Vector{T},
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

# Phase2のループの中の部分
function phase2_jump(A::Matrix{T}, Wᶜ::Matrix{T}, k::Int, n::Int)::T where {T <: Real}
    ε = 1e-8 # << 1

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        @variable(model, l[i=1:n] ≥ ε)
        
        for j = 1:n
            for i = filter(i -> i != j, 1:n)
                aᵢⱼ = A[i,j]
                wᵢᶜ = Wᶜ[i,k]; wⱼᶜ = Wᶜ[j,k]
                lᵢ = l[i]; lⱼ = l[j]
                @constraint(model, aᵢⱼ*(wⱼᶜ-lⱼ) ≤ wᵢᶜ+lᵢ)
            end
        end

        for j = 1:n
            lⱼ = l[j];
            wⱼᶜ = Wᶜ[j,k];
            # Sᵁ = Σ(μₖ*wᵢᶜ+lᵢ)
            # Sᴸ = Σ(μₖ*wᵢᶜ-lᵢ)
            Sᵁ = sum(map(i -> Wᶜ[i,k]+l[i], filter(i -> i != j, 1:n)))
            Sᴸ = sum(map(i -> Wᶜ[i,k]-l[i], filter(i -> i != j, 1:n)))
            @constraint(model, Sᵁ + wⱼᶜ-lⱼ ≥ 1)
            @constraint(model, Sᴸ + wⱼᶜ+lⱼ ≤ 1)
        end

        for i = 1:n
            wᵢᶜ = Wᶜ[i,k]; lᵢ = l[i]
            @constraint(model, wᵢᶜ - lᵢ ≥ ε)
        end

        dₖ  = sum(map(j -> l[j], filter(j -> j != k, 1:n)))
        @objective(model, Min, dₖ)

        optimize!(model)
        d̂ₖ = sum(map(j -> value.(l[j]), filter(j -> j != k, 1:n)))
        
        return d̂ₖ
        
    finally
        empty!(model)
    end
end

# Phase3のループの中の部分
function phase3_jump(A::Matrix{T}, Wᶜ::Matrix{T}, d̂::T, k::Int, n::Int)::phase3_jump_result{T} where {T <: Real}
    ε = 1e-8 # << 1

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        @variable(model, l[i=1:n] ≥ ε)
        lₖ = l[k]
        
        for j = 1:n
            for i = filter(i -> i != j, 1:n)
                aᵢⱼ = A[i,j]
                wᵢᶜ = Wᶜ[i,k]; wⱼᶜ = Wᶜ[j,k]
                lᵢ = l[i]; lⱼ = l[j]
                @constraint(model, aᵢⱼ*(wⱼᶜ-lⱼ) ≤ wᵢᶜ+lᵢ)
            end
        end

        for j = 1:n
            lⱼ = l[j];
            wⱼᶜ = Wᶜ[j,k];
            # Sᵁ = Σ(μₖ*wᵢᶜ+lᵢ)
            # Sᴸ = Σ(μₖ*wᵢᶜ-lᵢ)
            Sᵁ = sum(map(i -> Wᶜ[i,k]+l[i], filter(i -> i != j, 1:n)))
            Sᴸ = sum(map(i -> Wᶜ[i,k]-l[i], filter(i -> i != j, 1:n)))
            @constraint(model, Sᵁ + wⱼᶜ-lⱼ ≥ 1)
            @constraint(model, Sᴸ + wⱼᶜ+lⱼ ≤ 1)
        end

        for i = 1:n
            wᵢᶜ = Wᶜ[i,k]; lᵢ = l[i]
            @constraint(model, wᵢᶜ - lᵢ ≥ ε)
        end

        Σl = sum(map(j-> l[j], filter(j -> j != k, 1:n)))
        @constraint(model, Σl ≤ d̂+ε)

        @objective(model, Min, lₖ)

        optimize!(model)
        l̂ = value.(l) 
        return l̂

    finally
        empty!(model)
    end
end

# 提案手法
function MMR_W(A::Matrix{T}, method::Function)::LPResult_Individual{T} where {T <: Real}

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    m, n = size(A)

    # Phase 1
    # EV, GMで区間重要度の中心を求める
    Wᶜ = method(A)

    # Phase 2
    d̂ = Vector{T}(undef, n) 
    for k = 1:n
        d̂[k] = phase2_jump(A, 
        Wᶜ, k, n)
    end

    # Phase 3
    wᴸ = Matrix{T}(undef, m, n)
    wᵁ = Matrix{T}(undef, m, n)
    for k = 1:n
        l̂ = phase3_jump(A, Wᶜ, d̂[k], k, n)

        for i = 1:n
            l̂ᵢ = l̂[i]
            wᵢᶜ = Wᶜ[i,k]

            wᴸᵢ = wᵢᶜ - l̂ᵢ
            wᵁᵢ = wᵢᶜ + l̂ᵢ

            # 各kの時の推定値を縦ベクトルとして格納
            wᴸ[i, k] = wᴸᵢ
            wᵁ[i, k] = wᵁᵢ
        end
    end

    # Phase 4
    w̅̅ᴸ = Vector{T}(undef, n)
    w̅̅ᵁ = Vector{T}(undef, n)
    W̅̅ = Vector{Interval{T}}(undef, n)

    for i = 1:n
        w̅̅ᴸ[i] = minimum(wᴸ[i, :])
        w̅̅ᵁ[i] = maximum(wᵁ[i, :])

        # precision error 対応
        if w̅̅ᴸ[i] > w̅̅ᵁ[i]
            w̅̅ᴸ[i] = w̅̅ᵁ[i]
        end
        
        W̅̅[i] = (w̅̅ᴸ[i])..(w̅̅ᵁ[i])
    end

    return (
        wᴸ=w̅̅ᴸ, wᵁ=w̅̅ᵁ,
        W=W̅̅
    )

end