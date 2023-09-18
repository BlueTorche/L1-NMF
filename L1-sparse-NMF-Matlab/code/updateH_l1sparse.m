function H = updateH_l1sparse(W,H,cols)
    [r,n] = size(H);
    sumW  = sum(W);
    
    % Update de H
    for k = 1 : n
        indX  = cols{1,k};
        valX  = cols{2,k};
        Wt    = W(indX,:);
        WtH   = Wt * H(:,k);
%         flag=0;
%         if max(WtH)>1
%             flag=1;
%             Wt
%             H(:,k)
%             sum(H(:,k))
%             WtH
%         end
        
        sumWt = sum(Wt,1);
        for i = 1 : r
            Hikold   = H(i,k);
            valseuil = sumWt(i)-sumW(i)/2;
            if valseuil>eps
                y         = Wt(:,i);
                indi      = abs(y) > 1e-16; % Reduce the problem for nonzero entries of y
                y         = y.*indi;
                A         = valX - WtH + y * Hikold;
                tem=valX + y * Hikold;
%                 if max(tem)>1 && max(WtH)>1
%                     [tem -WtH]
%                 end
%                 if flag==1
%                 [valX -WtH y*Hikold A y A.*indi A.*indi./y]
%                 end
                A         = A.*indi;
                
                A         = A./y;
                %A
                [As,Inds] = sort(A); % Sort rows of A, O(n log(n)) operations
                s         = cumsum(y(Inds));
                H(i,k)    = max(0,As(find(s>=valseuil,1)));
                %if flag==1
%                 if H(i,k)>1
%                     H(i,k)
%                 end
            else
                H(i,k) = 0;
            end
            WtH = WtH + Wt(:,i)*(H(i,k)-Hikold);
        end
    end
end