function global_lambda_l1_norm_loss(X, W, H, lambda)
    WH = W * H
    n, m = size(X)
    total_errors = 0.0
    
    for i in 1:n
        for j in 1:m
            if X[i, j] == 0
                total_errors += lambda * abs(WH[i, j])
            else
                total_errors += abs(X[i, j] - WH[i, j])
            end
        end
    end

    return total_errors / (n * m)
end


function local_lambda_l1_norm_loss(X, W, H, lambdaH)
    WH = W*H
    m,n = size(X)
    errors = 0
    for i in 1:m
        for j in 1:n
            if X[i,j] == 0
                errors += lambdaH[j]*WH[i,j]
            else
                errors += abs(X[i,j]-WH[i,j])
            end
        end
    end

    return errors/(n*m)
end


function l2_norm_loss(X, W, Ht)
    return sum(X.*X) - 2*sum(W.*(X*Ht)) + sum((W'*W).*(Ht'*Ht))
end


function normalize(W)
    m,r = size(W)
    for p in 1:r
        col_max = maximum(W[:, p])
        if col_max > 0
            W[:, p] ./= col_max
        end
    end
    return W
end 


function find_cols_null(X::AbstractMatrix{T}) where T <: AbstractFloat
    cols_null = []
    _, n = size(X)
    
    for q in 1:n
        indices = findall(x -> x > 0, X[:, q])
        if typeof(indices) != Vector{Int}
            indices = [indices]
        end
        push!(cols_null, indices)
    end
    
    return cols_null
end