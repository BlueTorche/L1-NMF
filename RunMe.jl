using L1NMF

function preprossessing(X)
    m,_ = size(X)
    for p in 1:m
        somme = length(findall(x -> x > 0, X[p,:]))
        X[p,:] *= log(m/somme)
    end
    return X
end


X  = [[1.0 2.0 3.0][4.0 5.0 6.0][7.0 8.0 9.0]]
H = [1.0 1.0 1.0]
W = [[1.0][1.0][1.0]]
r = 1

W,H,_,_ = L1NMF.l1_sparse_nmf(X,r,W0=W,H0=H)

