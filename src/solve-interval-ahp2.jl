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

function phase1_test(A::Matrix{T}, method::Function)::Matrix{T} where {T <: Real}

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

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

function phase2_test(A::Matrix{T}, method::Function)::Vector{T} where {T <: Real}
    ε = 1e-8 # << 1

    Wᶜ = phase1_test(A, method) # phase1の動作は確認済
    m, n = size(A)
    # modelの定義はループの外に置いていても大丈夫なのか？ループの中に入れるべき？
   
    # Phase 2
    d⃰ = Vector{T}(undef, n) 
    for k = 1:n

        model = Model(HiGHS.Optimizer)
        set_silent(model)

        try
            @variable(model, l[i=1:n] ≥ ε)
            @variable(model, ε<=μₖ<=1-ε)
            lₖ = l[k]
            
            for j = filter(j -> j != k, 1:n)
                for i = filter(i -> i != j, 1:n)
                    aᵢⱼ = A[i,j]
                    wᵢᶜ = Wᶜ[i,k]; wⱼᶜ = Wᶜ[j,k]
                    lᵢ = l[i]; lⱼ = l[j]
                    @constraint(model, aᵢⱼ*(μₖ*wⱼᶜ-lⱼ) ≤ μₖ*wᵢᶜ+lᵢ)
                end
            end

            for i = filter(i -> i != k, 1:n)
                aᵢₖ = A[i,k]
                lᵢ = l[i]
                wᵢᶜ = Wᶜ[i,k]
                @constraint(model, aᵢₖ*(1-μₖ-lₖ) ≤ μₖ*wᵢᶜ+lᵢ)
                @constraint(model, μₖ*wᵢᶜ - lᵢ ≥ ε)
            end

            for j = filter(j -> j != k, 1:n)
                lⱼ = l[j]; lₖ = l[k]
                wⱼᶜ = Wᶜ[j,k];
                # Sᵁ = Σ(wᵢᶜ+lᵢ)
                # Sᴸ = Σ(wᵢᶜ-lᵢ)
                Sᵁ = sum(map(i -> Wᶜ[i,k]+l[i], filter(i -> i != j && i != k, 1:n)))
                Sᴸ = sum(map(i -> Wᶜ[i,k]-l[i], filter(i -> i != j && i != k, 1:n)))
                @constraint(model, (1-μₖ)+lₖ + Sᵁ + μₖ*wⱼᶜ-lⱼ ≥ 1)
                @constraint(model, (1-μₖ)-lₖ + Sᴸ + μₖ*wⱼᶜ+lⱼ ≤ 1)
            end

            # Sᵁ_dash = Σ(wᵢᶜ+lᵢ)
            # Sᴸ_dash = Σ(wᵢᶜ-lᵢ)
            Sᵁ_dash = sum(map(i -> Wᶜ[i,k]+l[i], filter(i -> i != k, 1:n)))
            Sᴸ_dash = sum(map(i -> Wᶜ[i,k]-l[i], filter(i -> i != k, 1:n)))
            @constraint(model, Sᵁ_dash + (1-μₖ)-lₖ ≥ 1)
            @constraint(model, Sᴸ_dash + (1-μₖ)+lₖ ≤ 1)
            @constraint(model, (1-μₖ)-lₖ ≥ ε)

            dₖ  = sum(map(j -> l[j], filter(j -> j != k, 1:n)))
            @objective(model, Min, dₖ)

            optimize!(model)
            dₖ⃰ = sum(map(j -> value.(l[j]), filter(j -> j != k, 1:n)))
            d⃰[k] = dₖ⃰
            
        finally
            empty!(model)
        end  
    end

    return d⃰
end

function solveIntervalAHP(A::Matrix{T}, method::Function)::LPResult_Individual{T} where {T <: Real}
# HACK:長すぎる。関数に切り分け
    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    ε = 1e-8 # << 1
    m, n = size(A)

    # Phase 1
    Wᶜ = Matrix{T}(undef, m, n) # 各kで見積もった重要度中心を格納
    for k = 1:n
        removed_matrix = remove_row_col(A, k, k)
        W = method(removed_matrix)

        # Wᶜₖは使わないが、ひとまず1として挿入
        insert!(W, k, 1.0)
        Wᶜ[:, k] = W
    end

    # Phase 2
    # FIXME:どの手法でも第1要素以外が0になる（CI=0の行列において）
    d⃰ = Vector{T}(undef, n) # 各kでの最適値を格納
    for k = 1:n

        model = Model(HiGHS.Optimizer)
        set_silent(model)

        try
            @variable(model, l[i=1:n] ≥ ε)
            @variable(model, ε<=μₖ<=1-ε)
            lₖ = l[k]
            
            for j = filter(j -> j != k, 1:n)
                for i = filter(i -> i != j, 1:n)
                    aᵢⱼ = A[i,j]
                    wᵢᶜ = Wᶜ[i,k]; wⱼᶜ = Wᶜ[j,k]
                    lᵢ = l[i]; lⱼ = l[j]
                    @constraint(model, aᵢⱼ*(μₖ*wⱼᶜ-lⱼ) ≤ μₖ*wᵢᶜ+lᵢ)
                end
            end

            for i = filter(i -> i != k, 1:n)
                aᵢₖ = A[i,k]
                lᵢ = l[i]
                wᵢᶜ = Wᶜ[i,k]
                @constraint(model, aᵢₖ*(1-μₖ-lₖ) ≤ μₖ*wᵢᶜ+lᵢ)
                @constraint(model, μₖ*wᵢᶜ - lᵢ ≥ ε)
            end

            for j = filter(j -> j != k, 1:n)
                lⱼ = l[j]; lₖ = l[k]
                wⱼᶜ = Wᶜ[j,k];
                # Sᵁ = Σ(wᵢᶜ+lᵢ)
                # Sᴸ = Σ(wᵢᶜ-lᵢ)
                Sᵁ = sum(map(i -> Wᶜ[i,k]+l[i], filter(i -> i != j && i != k, 1:n)))
                Sᴸ = sum(map(i -> Wᶜ[i,k]-l[i], filter(i -> i != j && i != k, 1:n)))
                @constraint(model, (1-μₖ)+lₖ + Sᵁ + μₖ*wⱼᶜ-lⱼ ≥ 1)
                @constraint(model, (1-μₖ)-lₖ + Sᴸ + μₖ*wⱼᶜ+lⱼ ≤ 1)
            end

            # Sᵁ_dash = Σ(wᵢᶜ+lᵢ)
            # Sᴸ_dash = Σ(wᵢᶜ-lᵢ)
            Sᵁ_dash = sum(map(i -> Wᶜ[i,k]+l[i], filter(i -> i != k, 1:n)))
            Sᴸ_dash = sum(map(i -> Wᶜ[i,k]-l[i], filter(i -> i != k, 1:n)))
            @constraint(model, Sᵁ_dash + (1-μₖ)-lₖ ≥ 1)
            @constraint(model, Sᴸ_dash + (1-μₖ)+lₖ ≤ 1)
            @constraint(model, (1-μₖ)-lₖ ≥ ε)

            dₖ  = sum(map(j -> l[j], filter(j -> j != k, 1:n)))
            @objective(model, Min, dₖ)

            optimize!(model)
            dₖ⃰ = sum(map(j -> value.(l[j]), filter(j -> j != k, 1:n)))
            d⃰[k] = dₖ⃰
            
        finally
            empty!(model)
        end  
    end

    # Phase 3
    # FIXME:ADのときのみwᴸ=0
    wᴸ = Matrix{T}(undef, m, n)
    wᵁ = Matrix{T}(undef, m, n)
    for k = 1:n
        model = Model(HiGHS.Optimizer)
        set_silent(model)

        try
            @variable(model, l[i=1:n] ≥ ε)
            @variable(model, ε<=μₖ<=1-ε)
            lₖ = l[k]
            
            for j = filter(j -> j != k, 1:n)
                for i = filter(i -> i != j, 1:n)
                    aᵢⱼ = A[i,j]
                    wᵢᶜ = Wᶜ[i,k]; wⱼᶜ = Wᶜ[j,k]
                    lᵢ = l[i]; lⱼ = l[j]
                    @constraint(model, aᵢⱼ*(μₖ*wⱼᶜ-lⱼ) ≤ μₖ*wᵢᶜ+lᵢ)
                end
            end

            for i = filter(i -> i != k, 1:n)
                aᵢₖ = A[i,k]
                lᵢ = l[i]
                wᵢᶜ = Wᶜ[i,k]
                @constraint(model, aᵢₖ*(1-μₖ-lₖ) ≤ μₖ*wᵢᶜ+lᵢ)
                @constraint(model, μₖ*wᵢᶜ - lᵢ ≥ ε)
            end

            for j = filter(j -> j != k, 1:n)
                lⱼ = l[j]; lₖ = l[k]
                wⱼᶜ = Wᶜ[j,k];
                # Sᵁ = Σ(wᵢᶜ+lᵢ)
                # Sᴸ = Σ(wᵢᶜ-lᵢ)
                Sᵁ = sum(map(i -> Wᶜ[i,k]+l[i], filter(i -> i != j && i != k, 1:n)))
                Sᴸ = sum(map(i -> Wᶜ[i,k]-l[i], filter(i -> i != j && i != k, 1:n)))
                @constraint(model, (1-μₖ)+lₖ + Sᵁ + μₖ*wⱼᶜ-lⱼ ≥ 1)
                @constraint(model, (1-μₖ)-lₖ + Sᴸ + μₖ*wⱼᶜ+lⱼ ≤ 1)
            end

            # Sᵁ_dash = Σ(wᵢᶜ+lᵢ)
            # Sᴸ_dash = Σ(wᵢᶜ-lᵢ)
            Sᵁ_dash = sum(map(i -> Wᶜ[i,k]+l[i], filter(i -> i != k, 1:n)))
            Sᴸ_dash = sum(map(i -> Wᶜ[i,k]-l[i], filter(i -> i != k, 1:n)))
            @constraint(model, Sᵁ_dash + (1-μₖ)-lₖ ≥ 1)
            @constraint(model, Sᴸ_dash + (1-μₖ)+lₖ ≤ 1)
            @constraint(model, (1-μₖ)-lₖ ≥ ε)
            Σl = sum(map(j-> l[j], filter(j -> j != k, 1:n)))
            @constraint(model, Σl == d⃰[k])

            dₖ  = sum(map(j -> l[j], filter(j -> j != k, 1:n)))
            @objective(model, Min, l[k])

            optimize!(model)
            μₖ⃰ = value.(μₖ)
            l⃰ = value.(l)
            
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

        finally
            empty!(model)
        end  
    end

    # Phase 4
    w̅̅ᴸ = Vector{T}(undef, n)
    w̅̅ᵁ = Vector{T}(undef, n)
    W̅̅ = Vector{Interval{T}}(undef, n)

    for i = 1:n
        w̅̅ᴸ[i] = minimum(wᴸ[i, :])
        w̅̅ᵁ[i] = maximum(wᵁ[i, :])
        W̅̅[i] = (w̅̅ᴸ[i])..(w̅̅ᵁ[i])
    end

    return (
        wᴸ=w̅̅ᴸ, wᵁ=w̅̅ᵁ,
        W=W̅̅
    )


end

# # 提案手法
# # methodにはAD, EV, GMを指定
# function solveIntervalAHP(A::Matrix{T}, method::Function)::LPResult_Individual{T} where {T <: Real}
#     ε = 1e-8 # << 1

#     if !isCrispPCM(A)
#         throw(ArgumentError("A is not a crisp PCM"))
#     end

#     m, n = size(A)
#     model = Model(HiGHS.Optimizer)
#     set_silent(model)

#     wᴸ = Matrix{T}(undef, m, n) 
#     wᵁ = Matrix{T}(undef, m, n) 
#     try
#         # Phase 1
#         Wᶜ = Matrix{T}(undef, m, n) 
#         for k = 1:n
#             removed_matrix = remove_row_col(A, k, k)
#             W = method(removed_matrix)

#             # Wᶜₖを1として挿入
#             insert!(W, k, 1.0)
#             Wᶜ[:, k] = W # k番目の推定データを縦ベクトルとして格納
#         end
#         print("phase1")

#         # Phase 2
#         for k = 1:n
#             @variable(model, l[i=1:n] ≥ ε)
#             @variable(model, ε<=μₖ<=1-ε)
#             lₖ = l[k]
            
#             for j = filter(j -> j != k, 1:n)
#                 for i = filter(i -> i != j, 1:n)
#                     aᵢⱼ = A[i,j]
#                     wᵢᶜ = Wᶜ[i,k]; wⱼᶜ = Wᶜ[j,k]
#                     lᵢ = l[i]; lⱼ = l[j]
#                     @constraint(model, aᵢⱼ*(μₖ*wⱼᶜ-lⱼ) ≤ μₖ*wᵢᶜ+lᵢ)
#                 end
#             end

#             for i = filter(i -> i != k, 1:n)
#                 aᵢₖ = A[i,k]
#                 lᵢ = l[i]
#                 wᵢᶜ = Wᶜ[i,k]
#                 @constraint(model, aᵢₖ*(1-μₖ-lₖ) ≤ μₖ*wᵢᶜ+lᵢ)
#                 @constraint(model, μₖ*wᵢᶜ - lᵢ ≥ ε)
#             end

#             for j = filter(j -> j != k, 1:n)
#                 lⱼ = l[j]; lₖ = l[k]
#                 wⱼᶜ = Wᶜ[j,k];
#                 # Sᵁ = Σ(wᵢᶜ+lᵢ)
#                 # Sᴸ = Σ(wᵢᶜ-lᵢ)
#                 Sᵁ = sum(map(i -> Wᶜ[i,k]+l[i], filter(i -> i != j && i != k, 1:n)))
#                 Sᴸ = sum(map(i -> Wᶜ[i,k]-l[i], filter(i -> i != j && i != k, 1:n)))
#                 @constraint(model, (1-μₖ)+lₖ + Sᵁ + μₖ*wⱼᶜ-lⱼ ≥ 1)
#                 @constraint(model, (1-μₖ)-lₖ + Sᴸ + μₖ*wⱼᶜ+lⱼ ≤ 1)
#             end

#             # Sᵁ_dash = Σ(wᵢᶜ+lᵢ)
#             # Sᴸ_dash = Σ(wᵢᶜ-lᵢ)
#             Sᵁ_dash = sum(map(i -> Wᶜ[i,k]+l[i], filter(i -> i != k, 1:n)))
#             Sᴸ_dash = sum(map(i -> Wᶜ[i,k]-l[i], filter(i -> i != k, 1:n)))
#             @constraint(model, Sᵁ_dash + (1-μₖ)-lₖ ≥ 1)
#             @constraint(model, Sᴸ_dash + (1-μₖ)+lₖ ≤ 1)
#             @constraint(model, (1-μₖ)-lₖ ≥ ε)


#             dₖ  = sum(map(j -> l[j], filter(j -> j != k, 1:n)))
#             @objective(model, Min, dₖ)

#             optimize!(model)
#             dₖ⃰ = sum(map(j -> l[j], filter(j -> j != k, 1:n)))
#         end
#         print("phase2")

#         # Phase 3
#         μ_star = T[]
#         l_star = Matrix{T}(undef, m, n)
#         for k = 1:n
#             @variable(model, l[i=1:n] ≥ ε)
#             @variable(model, 0<=μₖ<=1)
            
#             d_kbar = 0 # 変数名なおす
#             for j = filter(j -> k != j, 1:n)
#                 sum_plus = 0
#                 sum_minus = 0
#                 for i = filter(i -> j != i, 1:n)
#                     aᵢⱼ = A[i,j]
#                     print("phase2-1-3")
#                     @constraint(model, aᵢⱼ*(μₖ*Wᶜ[j,k]-l[j]) ≤ μₖ*Wᶜ[i,k]+l[i])
#                     print("phase2-2")

#                     sum_plus1 += μₖ*Wᶜ[i,k] + l[i]
#                     sum_minus1 += μₖ*Wᶜ[i,k] - l[i]
#                 end
#                 @constraint(model, (1-μₖ) + l[k] + sum_plus1 + μₖ*Wᶜ[j,k] - l[j] ≥ 1)
#                 print("phase2-3")

#                 @constraint(model, (1-μₖ) - l[k] + sum_minus1 + μₖ*Wᶜ[j,k] + l[j] ≤ 1)
#                 print("phase2-4")

#                 @constraint(model, μₖ*Wᶜ[j,k] - l[j] ≥ ε)
#                 print("phase2-5")
                
#                 if j != k
#                     d_kbar = l[j]
#                 end
#             end
#             sum_plus2 = 0
#             sum_plus2 = 0
#             for i = filter(i -> k != i, 1:n)
#                 aᵢₖ = A[i,k]
#                 sum_plus2 += μₖ*Wᶜ[i,k] + l[i]
#                 sum_minus2 += μₖ*Wᶜ[i,k] - l[i]
#                 @constraint(model, aᵢⱼ*((1-μₖ)-l[k]) ≤ μₖ*Wᶜ[i,k]+l[i])
#                 @constraint(model, sum_plus2 + (1-μₖ) - l[k] ≥ 1)
#                 @constraint(model, sum_minus2 + (1-μₖ) + l[k] ≤ 1)

#             end
#             @constraint(model, aᵢₖ*(μₖ*Wᶜ[j,k]-l[j]) ≤ μₖ*Wᶜ[i,k]+l[i])
#             @constraint(model, (1-μₖ) - l[k] ≥ ε)
#             @objective(model, Min, d_kbar)
#             optimize!(model)
#             d_star[k] = sum(map(i -> l[i], filter(i -> k != i, 1:n)))


#             for i = 1:n
#                 wᴸ[i, k] = μ_star*Wᶜ[i,k] - l_star[i]
#                 wᵁ[i, k] = μ_star*Wᶜ[i,k] + l_star[i]
#             end
#         end

#         # 最適値
#         for i = 1:n
#             wᴸ_barbar[i] = min(wᴸ[i, :])
#             wᵁ_barbar[i] = max(Wᵁ[i, :])
#         end
#         W_value = map(i -> (wᴸ_barbar[i])..(wᵁ_barbar[i]), 1:n)

#         return (
#             wᴸ=wᴸ_barbar, wᵁ=wᵁ_barbar,
#             W=W_value
#         )
#     finally
#         # エラー終了時にも変数などを消去する
#         empty!(model)
#     end
# end