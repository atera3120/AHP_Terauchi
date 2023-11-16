using DataFrames, CSV
include("./crisp-pcm.jl")
include("./solve-interval-ahp.jl")

Estimation_Result = @NamedTuple{
    # 区間重みベクトル
    wᴸ::Vector{T}, wᵁ::Vector{T},
    W::Vector{Interval{T}}, # ([wᵢᴸ, wᵢᵁ])
} where {T <: Real}

function read_csv()
    # CSVファイルを読み込む
    df = CSV.File("./src/data/N5_a3_A_PCM_int.csv", header=false) |> DataFrame
    subdfs = split_dataframe(df, 5)
    df2 = CSV.File("./src/data/N5_Given_interval_weight.csv", header=false) 
end

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

# データフレームを整形する関数
function reshape_dataframe(df, new_rows, new_cols)
    if nrow(df) * ncol(df) != new_rows * new_cols
        error("サイズが一致しません。")
    end
    return DataFrame(reshape(df, (new_rows, new_cols)))
end

