function [z,w,Dh,X,Y,Grr,Grs,Gss,Bl,Xr,Rx,Jac,Q,glo_num,M,dA]=set_sem(N,Nelx,Nely,xc,yc);
hdr

N1=N+1;
E=Nelx*Nely;

[z,w]=zwgll(N);                % Set basic operators
Dh=deriv_mat(z);

[R,S]=ndgrid(z,z);             % Build SEM mesh
X=zeros(N1,E,N1); Y=X;
e=0; 
for ey=1:Nely; for ex=1:Nelx; e=e+1;
    xe0=xc(ex); xe1=xc(ex+1); xeL=xe1-xe0;
    ye0=yc(ey); ye1=yc(ey+1); yeL=ye1-ye0;
    X(:,e,:) = xe0 + xeL*(R+1)/2;
    Y(:,e,:) = ye0 + yeL*(S+1)/2;
end; end;
[X,Y]=morph_circ(X,Y);         % Morph mesh

[Grr,Grs,Gss,Bl,Xr,Rx,Jac]=geom_elem(X,Y,Dh,w); % Terms for "A"
vol = sum(sum(sum(Bl)))
[Q,glo_num]=set_tp_semq(Nelx,Nely,N);
[M,Q,glo_num]=set_mask('P','P','D','N',N,Nelx,Nely,Q,glo_num); 
dA=diag_sem(Grr,Grs,Gss,Dh); dA=qqt(Q,dA); dA=1./dA; dA=M.*dA;

