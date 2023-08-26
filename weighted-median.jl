export weigthed_median

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