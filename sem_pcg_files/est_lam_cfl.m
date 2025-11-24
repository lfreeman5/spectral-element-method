function [lam_max_est]=est_lam_cfl(Cx,Cy,Rx);

% First, interpolate (Jac*geometry), which is polynomial

Cr = Rx(:,:,:,1,1).*Cx + Rx(:,:,:,1,2).*Cy;
Cs = Rx(:,:,:,2,1).*Cx + Rx(:,:,:,2,2).*Cy;

% Estimate | lambda_max |.   Recall: CFL = dt * max_i (Ui/dx_i)
N1=size(Cx,1); N=N1-1; E = size(Cx,2);
[z,w]=zwgll(N);
zd=diff(z);                  [dr,ds]=ndgrid(zd,zd);
zm=.5*(z(1:end-1)+z(2:end)); Jm=interp_mat(zm,z);
Crm = tensor3(Jm,1,Jm,Cr);
Csm = tensor3(Jm,1,Jm,Cs);
lam_max_est=-1e30;
for e=1:E;
   lam_m      = max(max(max( abs(Crm(:,e,:)./dr) +  abs(Csm(:,e,:)./ds) )));
   lam_max_est= max(lam_max_est,lam_m);
end;
%lam_max_est=1.3*lam_max_est; %% FROM DFM, Chapter 3.

