
function l1_with_sparsity_nmf(X::AbstractMatrix{T},
                        r::Integer;
                        sparsity::Integer=0,
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
                        alpha::Float64 = 0.8,
                        objfunction::Function = norml1_sparse,
                        benchmark::Bool = false,
                        reinstance::Bool = false,
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
        # println("End Initialization. Starting l1 NMF loop ...")
    end


    # Main NMF loop with error & time calc
    for it in 1:maxiter
        if benchmark
            # println("Iteration $it ...")
            time = @elapsed begin
                W = updaterW(X',H',W', WK, lambdaH; args...)'
                W = minmaxFilter(W')'
                H = updaterH(X, W, H, HK, lambdaH; args...)
                
                if reinstance
                    for i in 1:r
                        if sum(H[i,:]) == 0
                            W[:,i], H[i,:] = greedy_rank_one_l1nmf(X - W*H,HK,WK)
                        end
                    end
                end
            end

            push!(errors,objfunction(X, W, H, lambdaH))
            push!(times,time)
            
            if it > 1 && errors[it]-errors[it+1] < 10^-6
                println("End at Iteration $it")
                break
            end
        else
            W = updaterW(X',H',W', WK, lambdaH; args...)'
            W = minmaxFilter(W')'
            H = updaterH(X, W, H, HK, lambdaH; args...)

            if reinstance
                Z = copy(X) - W*H
                for i in 1:r
                    if sum(H[i,:]) == 0
                        W[:,i], H[i,:] = greedy_rank_one_l1nmf(Z,HK,WK)
                        Z -=  W[:,i]*H[i,:]'
                    end
                end
            end
        end

        # totSparsity = 0
        # if sparsity > 0
        #     nullsH = find_cols_null(H)
        #     for i in 1:m
        #         sparsityH = length(nullsH[i])
        #         if sparsityH > sparsity
        #             lambdaH[i] /= alpha^(sparsityH/sparsity)
        #         elseif sparsityH == 0
        #             lambdaH[i] *= alpha^(sparsityH/sparsity)
        #         end
        #         totSparsity += sparsityH
        #     end
        # end
        # totSparsity /= m
        # display(totSparsity)
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
            H[p,q] = constrained_weigthed_median(a, b)

            # Recalculate new value of v
            v = v + W[K,p]*(H[p,q]-Hold[p,q])
        end
    end 

    return H
end