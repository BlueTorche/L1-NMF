
function l1_with_sparsity_nmf(X::AbstractMatrix{T},
                        r::Integer;
                        lambdaH::AbstractArray{T}=zeros(T, 0, 0),
                        lambdaW::AbstractArray{T}=zeros(T, 0, 0),
                        maxiter::Integer = 20,
                        W0::AbstractMatrix{T} = zeros(T, 0, 0),
                        H0::AbstractMatrix{T} = zeros(T, 0, 0),
                        initializer::Function = nmf,
                        HK::AbstractMatrix{T} = zeros(T, 0, 0),
                        WK::AbstractMatrix{T} = zeros(T, 0, 0),
                        updaterH::Function = updateH_l1_sparse,
                        updaterW::Function = updateW_l1_sparse,
                        objfunction::Function = norml1_sparse,
                        benchmark::Bool = false,
                        args...
                        ) where T <: AbstractFloat
    if benchmark
        # println("Starting Initialization...")
        times = []
        errors = []
    end

    # Constants
    m, n = size(X)

    lambdaH =  length(lambdaH) == 0 ? ones(Float64, n) : lambdaH

    # If not provided, initiate W0 and H0
    if benchmark
        time = @elapsed begin
            W0, H0 = (length(W0) == 0 || length(H0) == 0) ? initializer(X,r,maxiter=5) : (W0, H0)
        end
        push!(times,time)
        push!(errors,objfunction(X, W0, H0, lambdaH))
    else
        W0, H0 = (length(W0) == 0 || length(H0) == 0) ? initializer(X,r,maxiter=5) : (W0, H0)
    end

    W, H = copy(W0), copy(H0)

    # Calculate index where X isn't 0 in each columns and in each rows
    HK = length(HK) == 0 ? find_cols_null(X) : HK
    WK = length(WK) == 0 ? find_cols_null(X') : WK

    if benchmark
        println("End Initialization. Starting l1 NMF loop ...")
    end


    # Main NMF loop with error & time calc
    for it in 1:maxiter
        if benchmark
            println("Iteration $it ...")
            time = @elapsed begin
                W = updaterW(X',H',W', WK, lambdaH; args...)'
                W = normalize(W')'
                H = updaterH(X, W, H, HK, lambdaH; args...)
            end

            push!(errors,objfunction(X, W, H, lambdaH))
            push!(times,time)
            
            if it > 1 && errors[it]-errors[it+1] < 10^-6
                println("End at Iteration $it")
                break
            end
        else
            W = updaterW(X',H',W', WK, lambdaH; args...)'
            W = normalize(W')'
            H = updaterH(X, W, H, HK, lambdaH; args...)
        end
    end

    if benchmark
        return W, H, times, errors
    end
    return W, H
end


function updateH_l1_sparse(X::AbstractMatrix{T},
                            W::AbstractMatrix{T},
                            H::AbstractMatrix{T},
                            KH::Vector{Any},
                            lambda::AbstractArray{T};
                            args...
                            ) where T <: AbstractFloat
    # Init
    r,n  = size(H)
    Hold = copy(H)

    # Calculate sums of rows of W
    S = Float64[]
    for p in 1:r
        push!(S,sum(W[:,p]))
    end

    # Loop on columns q
    for q in 1:n
        # Find index where X isn't 0 for each columns
        K = KH[q]

        # Calculate actual values H*K for the comlumn
        v = W[K,:]*H[:,q]

        # Loop on row r
        for p in 1:r
            # Calculate coefficient 'a', 'b' and 'c' of |X-WH|[q,p] = sum(|a_i-b_i*H[q,p]|) + |0-c*H[q,p]|
            b = W[K,p]
            a = X[K,q] - v + W[K,p]*H[p,q]
            c = S[p] - sum(W[K,p])

            # Apply tolerance lambda on c
            c *= lambda[q]

            # x = [a,0]
            push!(a,0.0)
            # y = [b,c]
            push!(b,c)

            # Calculate optimal value of H[p,q]
            H[p,q] = weigthed_median(a, b)

            # Recalculate new value of v
            v = v + W[K,p]*(H[p,q]-Hold[p,q])
        end
    end 

    return H
end

function updateW_l1_sparse(Xt::AbstractMatrix{T},
        Ht::AbstractMatrix{T},
        Wt::AbstractMatrix{T},
        KH::Vector{Any},
        lambda::AbstractArray{T};
        args...
        ) where T <: AbstractFloat
    # Init
    r,n  = size(Wt)
    Wtold = copy(Wt)

    # Calculate sums of rows of W
    S = Float64[]
    for p in 1:r
        push!(S,sum(Ht[:,p].*lambda))
    end

    # Loop on columns q
    for q in 1:n
        # Find index where X isn't 0 for each columns
        K = KH[q]

        # Calculate actual values H*K for the comlumn
        v = Ht[K,:]*Wt[:,q]

        # Loop on row r
        for p in 1:r
            # Calculate coefficient 'a', 'b' and 'c' of |X-WH|[q,p] = sum(|a_i-b_i*H[q,p]|) + |0-c*H[q,p]|
            b = Ht[K,p]
            a = Xt[K,q] - v + Ht[K,p]*Wt[p,q]
            c = S[p] - sum(Ht[K,p] .* lambda[K])

            # x = [a,0]
            push!(a,0.0)
            # y = [b,c]
            push!(b,c)

            # Calculate optimal value of H[p,q]
            Wt[p,q] = weigthed_median(a, b)

            # Recalculate new value of v
            v = v + Ht[K,p]*(Wt[p,q]-Wtold[p,q])
        end
    end 

    return Wt
end

function weigthed_median(x::AbstractVector{T},
                                     y::AbstractVector{T},
                                     args...
                                     ) where T <: AbstractFloat
    # Find index where y isn't null and only keep these values
    k = findall(x -> x != 0, y)
    y = y[k]
    x = x[k]

    # Return 0 if there is only 0
    if length(y) == 0
        return 0
    end

    # Calculate quotien x/y
    S = []
    for i in eachindex(y)
        push!(S,x[i]/y[i])
    end

    # Sort S and get the index to sort y the same way
    Inds = sortperm(S)
    S    = S[Inds]

    # Calculate born sum(y)/2
    # Find smallest index k such that sum_(i->k)y_i >= valseuil/2
    s        = cumsum(y[Inds])
    valseuil = last(s)/2
    k        = findall(x -> x >= valseuil,s)

    if length(k) == 0
        return 0
    end

    # Return value that will update H(p,q)
    return max(S[k[1]],0)
end