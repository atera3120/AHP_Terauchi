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

function phase1(A::Matrix{T}, method::Function)::Matrix{T} where {T <: Real}

    m, n = size(A)

    # Phase 1
    Wᶜ = Matrix{T}(undef, m, n) 
    for k = 1:n
        removed_matrix = remove_row_col(A, k, k)
        W = method(removed_matrix)

        # Wᶜₖを1として挿入
        insert!(W, k, 1.0)
        Wᶜ[:, k] = W
    end

    return Wᶜ
end

# Phase2のループの中の部分
function phase2_jump(A::Matrix{T}, Wᶜ::Matrix{T}, k::Int, n::Int)::T where {T <: Real}
    ε = 1e-6 # << 1

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        @variable(model, l[i=1:n] ≥ ε)
        @variable(model, ε<=μₖ<=1-ε)
        lₖ = l[k]
        
        for j = filter(j -> j != k, 1:n)
            for i = filter(i -> i != j && i != k, 1:n)
                aᵢⱼ = A[i,j]
                wᵢᶜ = Wᶜ[i,k]; wⱼᶜ = Wᶜ[j,k]
                lᵢ = l[i]; lⱼ = l[j]
                @constraint(model, aᵢⱼ*(μₖ*wⱼᶜ-lⱼ) ≤ μₖ*wᵢᶜ+lᵢ)
            end
        end

        for j = filter(j -> j != k, 1:n)
            aₖⱼ = A[k, j]
            lⱼ = l[j]
            wⱼᶜ = Wᶜ[j,k]
            @constraint(model, aₖⱼ*(μₖ*wⱼᶜ-lⱼ) ≤ 1-μₖ +lₖ)
        end

        for i = filter(i -> i != k, 1:n)
            aᵢₖ = A[i,k]
            lᵢ = l[i]
            wᵢᶜ = Wᶜ[i,k]
            @constraint(model, aᵢₖ*(1-μₖ-lₖ) ≤ μₖ*wᵢᶜ+lᵢ)
            @constraint(model, μₖ*wᵢᶜ - lᵢ ≥ ε)
        end

        for j = filter(j -> j != k, 1:n)
            lⱼ = l[j];
            wⱼᶜ = Wᶜ[j,k];
            # Sᵁ = Σ(μₖ*wᵢᶜ+lᵢ)
            # Sᴸ = Σ(μₖ*wᵢᶜ-lᵢ)
            Sᵁ = sum(map(i -> μₖ*Wᶜ[i,k]+l[i], filter(i -> i != j && i != k, 1:n)))
            Sᴸ = sum(map(i -> μₖ*Wᶜ[i,k]-l[i], filter(i -> i != j && i != k, 1:n)))
            @constraint(model, (1-μₖ)+lₖ + Sᵁ + μₖ*wⱼᶜ-lⱼ ≥ 1)
            @constraint(model, (1-μₖ)-lₖ + Sᴸ + μₖ*wⱼᶜ+lⱼ ≤ 1)
        end

        # Sᵁ_dash = Σ(μₖ*wᵢᶜ+lᵢ)
        # Sᴸ_dash = Σ(μₖ*wᵢᶜ-lᵢ)
        Sᵁ_dash = sum(map(i -> μₖ*Wᶜ[i,k]+l[i], filter(i -> i != k, 1:n)))
        Sᴸ_dash = sum(map(i -> μₖ*Wᶜ[i,k]-l[i], filter(i -> i != k, 1:n)))
        @constraint(model, Sᵁ_dash + (1-μₖ)-lₖ ≥ 1)
        @constraint(model, Sᴸ_dash + (1-μₖ)+lₖ ≤ 1)
        @constraint(model, (1-μₖ)-lₖ ≥ ε)

        dₖ  = sum(map(j -> l[j], filter(j -> j != k, 1:n)))
        @objective(model, Min, dₖ)

        optimize!(model)

        dₖ⃰ = sum(map(j -> value.(l[j]), filter(j -> j != k, 1:n)))

        # for i = 1:n
        #     println("k=$k, l[$i] = ", value.(l[i]))
        # end
        # println("k=$k, μₖ = ", value.(μₖ))
        # println("k=$k, dₖ⃰ = ", dₖ⃰)
        
        return dₖ⃰
        
    finally
        empty!(model)
    end
end

# Phase3の戻り値
phase3_jump_result = @NamedTuple{
    # 区間重みベクトル
    μₖ⃰::T, l⃰::Vector{T},
} where {T <: Real}

# Phase3のループの中の部分
function phase3_jump(A::Matrix{T}, Wᶜ::Matrix{T}, d⃰::T, k::Int, n::Int)::phase3_jump_result{T} where {T <: Real}
    ε = 1e-6 # << 1

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    try
        @variable(model, l[i=1:n] ≥ ε)
        @variable(model, ε<=μₖ<=1-ε)
        lₖ = l[k]
        
        for j = filter(j -> j != k, 1:n)
            for i = filter(i -> i != j && i != k, 1:n)
                aᵢⱼ = A[i,j]
                wᵢᶜ = Wᶜ[i,k]; wⱼᶜ = Wᶜ[j,k]
                lᵢ = l[i]; lⱼ = l[j]
                @constraint(model, aᵢⱼ*(μₖ*wⱼᶜ-lⱼ) ≤ μₖ*wᵢᶜ+lᵢ)
            end
        end

        for j = filter(j -> j != k, 1:n)
            aₖⱼ = A[k, j]
            lⱼ = l[j]
            wⱼᶜ = Wᶜ[j,k]
            @constraint(model, aₖⱼ*(μₖ*wⱼᶜ-lⱼ) ≤ 1-μₖ +lₖ)
        end

        for i = filter(i -> i != k, 1:n)
            aᵢₖ = A[i,k]
            lᵢ = l[i]
            wᵢᶜ = Wᶜ[i,k]
            @constraint(model, aᵢₖ*(1-μₖ-lₖ) ≤ μₖ*wᵢᶜ+lᵢ)
            @constraint(model, μₖ*wᵢᶜ - lᵢ ≥ ε)
        end

        for j = filter(j -> j != k, 1:n)
            lⱼ = l[j];
            wⱼᶜ = Wᶜ[j,k];
            # Sᵁ = Σ(μₖ*wᵢᶜ+lᵢ)
            # Sᴸ = Σ(μₖ*wᵢᶜ-lᵢ)
            Sᵁ = sum(map(i -> μₖ*Wᶜ[i,k]+l[i], filter(i -> i != j && i != k, 1:n)))
            Sᴸ = sum(map(i -> μₖ*Wᶜ[i,k]-l[i], filter(i -> i != j && i != k, 1:n)))
            @constraint(model, (1-μₖ)+lₖ + Sᵁ + μₖ*wⱼᶜ-lⱼ ≥ 1)
            @constraint(model, (1-μₖ)-lₖ + Sᴸ + μₖ*wⱼᶜ+lⱼ ≤ 1)
        end

        # Sᵁ_dash = Σ(μₖ*wᵢᶜ+lᵢ)
        # Sᴸ_dash = Σ(μₖ*wᵢᶜ-lᵢ)
        Sᵁ_dash = sum(map(i -> μₖ*Wᶜ[i,k]+l[i], filter(i -> i != k, 1:n)))
        Sᴸ_dash = sum(map(i -> μₖ*Wᶜ[i,k]-l[i], filter(i -> i != k, 1:n)))
        @constraint(model, Sᵁ_dash + (1-μₖ)-lₖ ≥ 1)
        @constraint(model, Sᴸ_dash + (1-μₖ)+lₖ ≤ 1)
        @constraint(model, (1-μₖ)-lₖ ≥ ε)
        Σl = sum(map(j-> l[j], filter(j -> j != k, 1:n)))
        @constraint(model, Σl ≤ d⃰+ε)

        # dₖ  = sum(map(j -> l[j], filter(j -> j != k, 1:n)))
        @objective(model, Min, l[k])

        optimize!(model)
        μₖ⃰ = value.(μₖ)
        l⃰ = value.(l) 
        return (
            μₖ⃰ = μₖ⃰ ,
            l⃰ = l⃰
        )

    finally
        empty!(model)
    end
end

# 提案手法 MMR-E, MMR-G, MMR-A
function AMR(A::Matrix{T}, method::Function)::LPResult_Individual{T} where {T <: Real}

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    m, n = size(A)

    # Phase 1
    Wᶜ = phase1(A, method)

    # Phase 2
    d⃰ = Vector{T}(undef, n) 
    for k = 1:n
        d⃰[k] = phase2_jump(A, Wᶜ, k, n)
    end

    # Phase 3
    wᴸ = Matrix{T}(undef, m, n)
    wᵁ = Matrix{T}(undef, m, n)
    for k = 1:n
        (μₖ⃰, l⃰) = phase3_jump(A, Wᶜ, d⃰[k], k, n)

        for i = 1:n
            lᵢ⃰ = l⃰[i]
            wᵢᶜ = Wᶜ[i,k]
            if i==k
                wᴸᵢ = (1-μₖ⃰ ) - lᵢ⃰
                wᵁᵢ = (1-μₖ⃰ ) + lᵢ⃰
            else
                wᴸᵢ = μₖ⃰ *wᵢᶜ - lᵢ⃰
                wᵁᵢ = μₖ⃰ *wᵢᶜ + lᵢ⃰
            end
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
        w̅̅ᴸ[i] = mean(wᴸ[i, :])
        w̅̅ᵁ[i] = mean(wᵁ[i, :])

        # precision error 対応
        if w̅̅ᴸ[i] > w̅̅ᵁ[i]
            w̅̅ᴸ[i] = w̅̅ᵁ[i]
        end
        
    end

    for i = 1:n
        W̅̅[i] = (w̅̅ᴸ[i])..(w̅̅ᵁ[i])
    end

    return (
        wᴸ=w̅̅ᴸ, wᵁ=w̅̅ᵁ,
        W=W̅̅
    )

end