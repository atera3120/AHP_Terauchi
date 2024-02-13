@inline function jakumodel(A::Matrix{T}, method::Function)::Vector{T} where {T <: Real}

    m, n = size(A)

    Wᶜ = Matrix{T}(undef, m, n) 
    for k = 1:n
        removed_matrix = remove_row_col(A, k, k)
        W = method(removed_matrix)

        # Wᶜₖを1として挿入
        insert!(W, k, 0)
        Wᶜ[:, k] = W
    end

    w = Vector{T}(undef, n)
    for i = 1:n
        w[i] = sum(Wᶜ[i,:])/n
    end
    return w
end

@inline function remove_row_col(A::Matrix{T}, row::Int, col::Int)::Matrix{T} where {T <: Real}
    m, n = size(A)

    # 行を除外
    new_matrix = A[setdiff(1:m, row), :]
    # 列を除外
    result_matrix = new_matrix[:, setdiff(1:n, col)]
        
    return result_matrix
end