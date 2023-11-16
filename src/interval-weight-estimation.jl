import Pkg; Pkg.add("DataFrame"); Pkg.add("CSV")
using DataFrame, CSV
include("./crisp-pcm.jl")
include("./solve-interval-ahp.jl")

Estimation_Result = @NamedTuple{
    # 区間重みベクトル
    wᴸ::Vector{T}, wᵁ::Vector{T},
    W::Vector{Interval{T}}, # ([wᵢᴸ, wᵢᵁ])
} where {T <: Real}

function read_csv()
    # CSVファイルを読み込む
    df = CSV.read("data.csv", DataFrame)
    # DataFrameをMatrixに変換
    A = Matrix(df)
    return A
end

function estimate_weight(A::Matrix{T}, method::Function)::Estimation_Result where {T <: Real}
    