using DataFrames
include("./crisp-pcm.jl")

# データフレームを分割する関数
@inline function split_dataframe(df, chunk_size)
    n = nrow(df)
    m = div(n, chunk_size)
    subdfs = []
    for i in 1:chunk_size:n
        push!(subdfs, df[i:min(i+chunk_size-1, n), :])
    end
    return subdfs
end

@inline function CI(A::Matrix{T})::T where {T <: Real}
    m, n = size(A)

    if !isCrispPCM(A)
        throw(ArgumentError("A is not a crisp PCM"))
    end

    λₘₐₓ = maximum(real(eigen(A).values))

    return CI = (λₘₐₓ - n) / (n - 1)

end

# Interval が空集合の場合は幅0を返す
@inline function c_diam(interval)
    if isempty(interval)
        return 0.0
    else
        return diam(interval)
    end
end

# P値
@inline function calculate_P(T, E)
    TcapE = T .∩ E
    TcupE = T .∪ E
    P = c_diam.(TcapE) ./ c_diam.(TcupE)
    return P
end

# Q値
@inline function calculate_Q(T, E)
    TcapE = T .∩ E
    Q = c_diam.(TcapE) ./ c_diam.(T)
    return Q
end

# R値
@inline function calculate_R(T, E)
    TcapE = T .∩ E
    R = Float64[]
    for i in eachindex(T)
        if c_diam(E[i])==0.0
            if issubset(E[i],T[i])
                push!(R, 1.0)
            else
                push!(R, 0.0)
            end
        else
            push!(R, c_diam(TcapE[i]) / c_diam(E[i]))
        end
    end
    # R = c_diam.(TcapE) ./ c_diam.(E)
    return R
end

# F値
@inline function calculate_F(T, E)
    Qv = calculate_Q(T, E)
    Rv = calculate_R(T, E)
    denominator = Qv .+ Rv
    # 分母が 0 でない場合のみ計算を行う
    F = ifelse.(denominator .== 0, 0.0, 2 * (Qv .* Rv) ./ denominator)
    return F
end

# crispな推定値がTの範囲に含まれているか
# 区間の中央値との距離
@inline function est_in_range(T, E)
    n = length(E)
    
    cnt = 0
    diff = 0.0

    for i in 1:n
        diff += ( (sup(T[i])+inf(T[i]))/2 -E[i] )^2
        if E[i] in T[i]
            cnt += 1
        end
    end
    euclid = sqrt(diff)
    
    return cnt, euclid
end
