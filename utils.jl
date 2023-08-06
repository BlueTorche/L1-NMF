function norml1(X, W, H, lambda)
    WH = W*H
    n,m = size(X)
    errors = 0
    for i in 1:n
        for j in 1:m
            if X[i,j] == 0
                errors += lambda*abs(WH[i,j])
            else
                errors += abs(X[i,j]-WH[i,j])
            end
        end
    end

    return errors/(n*m)
end

# Fast computation of reconstruction error
function fronorm(X, W, Ht)
    return sum(X.*X) - 2*sum(W.*(X*Ht)) + sum((W'*W).*(Ht'*Ht))
end


function normalize(W,r)
    for p in 1:r
        max = maximum(W[:,p])
        if max > 0
            W[:,p] /= max
        end
    end
    return W
end 


function find_cols_null(X::AbstractMatrix{T}) where T <: AbstractFloat
    cols_null = []
    _, n = size(X)
    for q in 1:n
        K = findall(x -> x > 0, X[:,q])
        push!(cols_null, typeof(K) != Vector{Int64} ? [K] : K)
    end
    return cols_null
end