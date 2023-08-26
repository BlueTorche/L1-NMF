export l2nmf

function l2nmf(X::AbstractMatrix{T},
             r::Integer;
             maxiter::Integer = 100,
             W0::AbstractMatrix{T} = zeros(T, 0, 0),
             H0::AbstractMatrix{T} = zeros(T, 0, 0),
             updaterW::Function = hals_updtW,
             updaterH::Function = hals_updtH,
             benchmark::Bool = false,
             objfunction::Function = l2_norm_loss,
             args...
             ) where T <: AbstractFloat
    # Constants
    m, n = size(X)

    # If not provided, init W and H randomly
    W0 = length(W0) == 0 ? rand(m, r) : W0
    W = copy(W0)
    H0 = length(H0) == 0 ? rand(r, n) : H0
    # Work on Ht to work along columns instead of rows (faster)
    Ht = copy(H0')

    # Main NMF loop
    times = zeros(Float64, maxiter)
    errors = zeros(Float64, maxiter)
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
                    XHt::AbstractMatrix{T}=zeros(T, 0, 0),
                    HHt::AbstractMatrix{T}=zeros(T, 0, 0);
                    args...
                    ) where T <: AbstractFloat
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
                    XtW::AbstractMatrix{T}=zeros(T, 0, 0),
                    WtW::AbstractMatrix{T}=zeros(T, 0, 0);
                    args...
                    ) where T <: AbstractFloat
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
