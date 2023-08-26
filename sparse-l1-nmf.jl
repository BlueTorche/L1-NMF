export l1_sparse_nmf

function l1_sparse_nmf(X::AbstractMatrix{T},
                     r::Integer;
                     lambda::Float64 = 1.0,
                     maxiter::Integer = 20,
                     W0::AbstractMatrix{T} = zeros(T, 0, 0),
                     H0::AbstractMatrix{T} = zeros(T, 0, 0),
                     initializer::Function = L1NMF.nmf,
                     HK::AbstractMatrix{T} = zeros(T, 0, 0),
                     WK::AbstractMatrix{T} = zeros(T, 0, 0),
                     updaterH::Function = updateH_l1_sparse,
                     updaterWt::Function = updateH_l1_sparse,
                     objfunction::Function = L1NMF.lamnda_norm_l1_loss,
                     benchmark::Bool = false,
                     reinstance::Bool = true,
                     args...
                     ) where T <: AbstractFloat

    if benchmark
        println("Starting Initialization...")
        times = []
        errors = []
    end

    if isnothing(updaterWt)
        updaterWt = updaterH
    end

    # Constants
    m, n = size(X)

    # If not provided, initiate W0 and H0
    if benchmark
        time = @elapsed begin
            W0, H0 = (length(W0) == 0 || length(H0) == 0) ? initializer(X,r,maxiter=5) : (W0, H0)
        end
        push!(times,time)
        push!(errors,objfunction(X, W0, H0, lambda))
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
                W = updaterWt(X',H',W', WK, lambda; args...)'
                W = normalize(W)
                H = updaterH(X, W, H, HK, lambda; args...)
            end

            push!(errors,objfunction(X, W, H, lambda))
            push!(times,time)
            
            if errors[it]-errors[it+1] < 10^-6
                println("End at Iteration $it")
                break
            end
        else
            W = updaterWt(X',H',W', WK, lambda; args...)'
            W = normalize(W)
            H = updaterH(X, W, H, HK, lambda; args...)
        end
    end

    if benchmark
        return W, H, times, errors 
    else 
        return W, H
    end
end


function updateH_l1_sparse(X::AbstractMatrix{T},
                           W::AbstractMatrix{T},
                           H::AbstractMatrix{T},
                           KH::Vector{Any},
                           lambda::Float64;
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
            c *= lambda

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