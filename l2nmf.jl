export l2nmf

function l2nmf(X::AbstractMatrix{T},
             r::Integer;
             maxiter::Integer       = 100,
             W0::AbstractMatrix{T}  = zeros(T, 0, 0),
             H0::AbstractMatrix{T}  = zeros(T, 0, 0),
             updaterW::Function     = hals_updtW,
             updaterH::Function     = hals_updtH,
             benchmark::Bool        = false,
             objfunction::Function  = l2_norm_loss,
             args...
             ) where T <: AbstractFloat
    """
        Perform Non-negative Matrix Factorization (NMF) using the L2-norm loss.

        This function factorizes the input matrix X into two non-negative matrices W and H
        such that X ≈ W * H. This is done by iteratively updating W and H using specified
        update functions until convergence or reaching the maximum number of iterations.

        # Arguments
            - X::AbstractMatrix{T}      : The input data matrix to be factorized of dimension (m x n).
            - r::Integer                : The rank of the factorization.
        # Optional Arguments
            - maxiter::Integer          : Maximum number of iterations for the factorization.   (default: 100)
            - W0::AbstractMatrix{T}     : Initial value for matrix W of dimension (m x r).      (default: zeros)
            - H0::AbstractMatrix{T}     : Initial value for matrix H of dimension (r x n).      (default: zeros)
            - updaterW::Function        : Function for updating matrix W.                       (default: hals_updtW)
            - updaterH::Function        : Function for updating matrix H.                       (default: hals_updtH)
            - benchmark::Bool           : Whether to benchmark the factorization process.       (default: false)
            - objfunction::Function     : Objective function for assessing convergence.         (default: l2_norm_loss)
            - args...                   : Additional arguments to be passed to the update functions.

        # Returns
            - W::AbstractMatrix{T}`     : Factorized matrix W.
            - H::AbstractMatrix{T}`     : Factorized matrix H.
        # Optional Returns
            - times::Vector{Float64}    : Array of execution times for each iteration. Returs only if benchmark == true.
            - errors::Vector{Float64}   : Array of loss function values for each iteration. Returs only if benchmark == true.
    """

    # Constants
    m, n = size(X)

    # If not provided, init W and H randomly
    W0 = length(W0) == 0 ? rand(m, r) : W0
    W = copy(W0)
    H0 = length(H0) == 0 ? rand(r, n) : H0
    # Work on Ht to work along columns instead of rows (faster)
    Ht = copy(H0')

    times = zeros(Float64, maxiter)
    errors = zeros(Float64, maxiter)

    # Main NMF loop
    for it in 1:maxiter
        if benchmark
            println("Iteration $it...")
            times[it] = @elapsed begin
                updaterW(X, W, Ht; args...)
                updaterH(X, W, Ht; args...)
            end
            errors[it] = objfunction(X, W, Ht)
        else
            updaterW(X, W, Ht; args...)
            updaterH(X, W, Ht; args...)
        end
    end
    if benchmark
        return W, Ht', times, errors
    else
        return W, Ht'
    end
end


function hals_updtW(X::AbstractMatrix{T},
                    W::AbstractMatrix{T},
                    Ht::AbstractMatrix{T},
                    XHt::AbstractMatrix{T} = zeros(T, 0, 0),
                    HHt::AbstractMatrix{T} = zeros(T, 0, 0);
                    args...
                    ) where T <: AbstractFloat
    """
    Perform Hierarchical  Alternating Least Squares (HALS) to update the matrix W.

    W =  W ◦ (X * H') / (H * H' * W) 

    # Arguments
        - X::AbstractMatrix{T}      : The input data matrix to be factorized.
        - W::AbstractMatrix{T}      : Actual value of the matrix W.
        - Ht::AbstractMatrix{T}     : Actual value of the matrix H transposed.
    # Optional Arguments
        - XHt::AbstractMatrix{T}    : Value of the product X * H'.      (default: zeros)
            If it has been compute before, can be passed to make the computation faster.
        - HHt::AbstractMatrix{T}    : Value of the product H * H'.      (default: zeros)
            If it has been compute before, can be passed to make the computation faster.
        - args...                   : Additional arguments that won't affect the function.
    """

    # Init
    r = size(W, 2)

    # If needed, compute intermediary values
    if length(XHt) == 0
        XHt = X * Ht
    end
    if length(HHt) == 0
        HHt = Ht' * Ht
    end

    # Loop on columns of W
    for j in 1:r
        jcolW = view(W, :, j)
        deltaW = max.((XHt[:,j] - W * HHt[:,j]) / max(HHt[j,j],1e-16), -jcolW)
        jcolW .+= deltaW
    end
end


function hals_updtH(X::AbstractMatrix{T},
                    W::AbstractMatrix{T},
                    Ht::AbstractMatrix{T},
                    XtW::AbstractMatrix{T} = zeros(T, 0, 0),
                    WtW::AbstractMatrix{T} = zeros(T, 0, 0);
                    args...
                    ) where T <: AbstractFloat
    """
    Perform Hierarchical  Alternating Least Squares (HALS) to update the matrix H.

    H' =  H' ◦ (X' * H) / (W' * W * H') 

    # Arguments
        - X::AbstractMatrix{T}      : The input data matrix to be factorized.
        - W::AbstractMatrix{T}      : Actual value of the matrix W.
        - Ht::AbstractMatrix{T}     : Actual value of the matrix H transposed.
    # Optional Arguments
        - XtW::AbstractMatrix{T}    : Value of the product X' * W.      (default: zeros)
            If it has been compute before, can be passed to make the computation faster.
        - WtW::AbstractMatrix{T}    : Value of the product W' * W.      (default: zeros)
            If it has been compute before, can be passed to make the computation faster.
        - args...                   : Additional arguments that won't affect the function.
    """
    # Init
    r = size(Ht, 2)

    # If needed, compute intermediary values
    if length(XtW) == 0
        XtW = X' * W
    end
    if length(WtW) == 0
        WtW = W' * W
    end

    # Loop on rows of H (columns of Ht)
    for i in 1:r
        irowH = view(Ht, :, i)
        deltaH = max.((XtW[:,i] - Ht * WtW[:,i]) /  max(WtW[i,i],1e-16), -irowH)
        irowH .+= deltaH
    end
end
