function [W,H] = NMFL1Sparse_arnaud(X,W,H,maxiter)
    if nargin==3
        maxiter = 100;
    end
    [m,n] = size(X);
    
    % Store the indices and values of the columns of X
    cols  = cell(2,n);
    for i=1:n
        [indX,~,valX] = find(X(:,i));
        cols{1,i} = indX;
        cols{2,i} = valX;
    end
    
    % Store the indices and values of the lines of X
    Xt = X';
    rows = cell(2,m);
    for i=1:m
        [indX,~,valX] = find(Xt(:,i));
        rows{1,i} = indX;
        rows{2,i} = valX;
    end

    % Perform Algorithm
    for iter=1:maxiter
%         iter
        Ht = H;
        Wt = W;    

        W = updateH_l1sparse(H',W',rows)'; %[Wt W]
        H = updateH_l1sparse(W,H,cols); %[Ht H]

        % Control the update of H and W to exit sooner if local minima
        deltah = sum(sum(abs(H-Ht)));
        deltaw = sum(sum(abs(W-Wt)));
        if deltah < 1e-16 && deltaw < 1e-16
            break
        end
    end

end

% function H = updateH_l1sparse(W,H,cols)
%     [r,n] = size(H);
%     sumW  = sum(W);
%     
%     % Update de H
%     for k = 1 : n
%         indX  = cols{1,k};
%         valX  = cols{2,k};
%         Wt    = W(indX,:);
%         WtH   = Wt * H(:,k);
%         sumWt = sum(Wt,1);
%         for i = 1 : r
%             Hikold   = H(i,k);
%             valseuil = sumWt(i)-sumW(i)/2;
%             if valseuil>eps
%                 y         = Wt(:,i);
%                 indi      = abs(y) > 1e-16; % Reduce the problem for nonzero entries of y
%                 y         = y.*indi;
%                 A         = valX - WtH + y * Hikold;
%                 A         = A.*indi;
%                 A         = A./y;
%                 [As,Inds] = sort(A); % Sort rows of A, O(n log(n)) operations
%                 s         = cumsum(y(Inds));
%                 H(i,k)    = max(0,As(find(s>=valseuil,1)));
%             else
%                 H(i,k) = 0;
%             end
%             WtH = WtH + Wt(:,i)*(H(i,k)-Hikold);
%         end
%     end
% end