using IntervalArithmetic

"""
行列を LaTeX 形式にする  
`L"matrixLaTeXString(A)"` とすると LaTeX 形式で表示できる
"""
function matrixLaTeXString(
    A::Matrix{T}
    )::String where {T <: Real}
    m, n = size(A)

    # 各成分の LaTeX 表記を入れる
    Aₛₜᵣ = fill("", (m,n))
    for i = 1:m, j = 1:n
        # 少数第4位で四捨五入
        Aₛₜᵣ[i,j] = string(round(A[i,j], digits=3))
    end

    str = "\\begin{pmatrix} "
    for i = 1:m, j = 1:n
        if j != n
            str = str * "$(Aₛₜᵣ[i,j]) & "
        else
            str = str * "$(Aₛₜᵣ[i,j]) \\\\"
        end
    end
    str = str * " \\end{pmatrix}"

    return str
end

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


"""
区間ベクトルを LaTeX 形式にする  
`L"intervalVectorLaTeXString(W)"` とすると LaTeX 形式で表示できる
"""
function intervalVectorLaTeXString(
        W::Vector{Interval{T}}
        )::String where {T <: Real}
    n = length(W)

    # 各成分の LaTeX 表記を入れる
    Wₛₜᵣ = fill("", n)
    for i = 1:n
        if iscommon(W[i])
            # (i,j)成分の両端
            # 少数第4位で四捨五入
            wᵢᴸ = string(round(W[i].lo, digits=3))
            wᵢᵁ = string(round(W[i].hi, digits=3))
            Wₛₜᵣ[i] = "\\left[ $(wᵢᴸ), $(wᵢᵁ) \\right]"
        else
            Wₛₜᵣ[i] = "\\emptyset"
        end
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