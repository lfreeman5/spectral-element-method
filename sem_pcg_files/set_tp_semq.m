function [Q,glo_num]=set_tp_semq(Nelx,Nely,N);

hdr;

% Set up Q for tensor-product array of Nth-order elements in 2D

N1 = N+1;
E  = Nelx*Nely;
nL = E*N1*N1;

glo_num=zeros(N1,E,N1);

e=0;

n=0;

for ey=1:Nely; for ex=1:Nelx; e=e+1;

     i0=1; if ex>1; i0=2; 
        glo_num(1,e,:) = glo_num(N1,e-1,:);
     end;

     j0=1; if ey>1; j0=2; 
        glo_num(:,e,1) = glo_num(:,e-Nelx,N1);
     end;

     for j=j0:N1; for i=i0:N1;
        n=n+1;
        glo_num(i,e,j) = n;
     end; end;

end; end;

% se_disp(glo_num,'glo_num')

Q=speye(nL); 
k=0;
for j=1:N1;
for e=1:E;
for i=1:N1;
    ig = glo_num(i,e,j);
    k  = k+1;
    Q(k,k)=0; Q(k,ig)=1;
end;
end;
end;

Q=Q(:,1:n);

