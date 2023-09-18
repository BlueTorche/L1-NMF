clear all; clc; close all;

m = 7;
n = 5;
r = 3;

X  = rand(m,n);
W0 = rand(m,r);
H0 = rand(r,n);

X(X>0.5)=1;
X(X<0.5)=0;

W0(W0>0.5)=1;
W0(W0<0.5)=0;

H0(H0>0.5)=1;
H0(H0<0.5)=0;

maxiter = 10;

%[W,H,e,Wt,Ht]=l1nmf_exact(X,W0,H0,maxiter);
%e

normX=sum(sum(abs(X)));

t0=cputime;
[Wex,Hex] = NMFL1Sparse_arnaud(X,W0,H0,maxiter);
texact=cputime-t0;
erexct=sum(sum(abs(X-W0*Hex)))/normX

t0=cputime;
[m,n] = size(X);
% Store the indices and values of the columns of X
cols  = cell(2,n);
for i=1:n
    [indX,~,valX] = find(X(:,i));
    cols{1,i} = indX;
    cols{2,i} = valX;
end
for tent=1:1000
  H0 = rand(r,n);
  H0(H0>0.5)=1;
  H0(H0<0.5)=0;
  Hhe=H0;
  nbiter = 100;
  for k=1:nbiter
      Hheprec = Hhe;
      Hhe = updateH_l1sparse(W0,Hhe,cols);
      ereiterH(k)=sum(sum(abs(Hhe-Hheprec)));
      eriterf(k)=sum(sum(abs(X-W0*Hhe)));
      tempsalgo(k) = cputime-t0;
      if sum(sum(abs(Hhe-Hheprec)))<1e-9
          break
      end
  end
  valtent(tent)=eriterf(end)/normX;
end
min(valtent)
normX=sum(sum(abs(X)));
plot(tempsalgo,eriterf/normX); hold on;

erexct=sum(sum(abs(X-W0*Hex)))/normX;
for k = 1:length(tempsalgo)
    if tempsalgo(k)>=texact
        eraxactf(k) = erexct;
    end
end
plot(tempsalgo,eraxactf)
sum(sum(abs(X-W0*Hhe)))