using DataFrames, CSV
include("./crisp-pcm.jl")
include("./solve-interval-ahp.jl")

# データフレームを分割する関数
function split_dataframe(df, chunk_size)
    n = nrow(df)
    m = div(n, chunk_size)
    subdfs = []
    for i in 1:chunk_size:n
        push!(subdfs, df[i:min(i+chunk_size-1, n), :])
    end
    return subdfs
end

# Interval が空集合の場合は幅0を返す
function c_diam(interval)
    if isempty(interval)
        return 0.0
    else
        return diam(interval)
    end
end

# P値
function calculate_P(T, E)
    TcapE = T .∩ E
    TcupE = T .∪ E
    P = c_diam.(TcapE) ./ c_diam.(TcupE)
    return P
end

# Q値
function calculate_Q(T, E)
    TcapE = T .∩ E
    Q = c_diam.(TcapE) ./ c_diam.(T)
    return Q
end

# R値
function calculate_R(T, E)
    TcapE = T .∩ E
    R = c_diam.(TcapE) ./ c_diam.(E)
    return R
end

# F値
function calculate_F(T, E)
    Qv = calculate_Q(T, E)
    Rv = calculate_R(T, E)
    denominator = Qv .+ Rv
    # 分母が 0 でない場合のみ計算を行う
    F = ifelse.(denominator .== 0, 0.0, 2 * (Qv .* Rv) ./ denominator)
    return F
end
