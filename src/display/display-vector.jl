using IntervalArithmetic

"""
重要度ベクトルを LaTeX 形式にする  
`L"VectorLaTeXString(W)"` とすると LaTeX 形式で表示できる
"""
function VectorLaTeXString(
        W::Vector{T}
        )::String where {T <: Real}
    n = length(W)

    # 各成分の LaTeX 表記を入れる
    Wₛₜᵣ = fill("", n)
    for i = 1:n
        # 少数第4位で四捨五入
        wᵢ = string(round(W[i], digits=3))
        Wₛₜᵣ[i] = wᵢ
    end

    str = "\\begin{pmatrix} "
    for i = 1:n
        str = str * "$(Wₛₜᵣ[i])"
        if i != n
            str = str * " \\\\ " # 改行
        end
    end
    str = str * " \\end{pmatrix}"

    return str
end