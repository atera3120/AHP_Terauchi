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