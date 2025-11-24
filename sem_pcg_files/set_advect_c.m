function [Cr,Cs]=set_advect_c(Cx,Cy,JM,BMh,Jac,Rx);

% First, interpolate (Jac*geometry), which is polynomial

Jrx = tensor3(JM,1,JM,Jac.*Rx(:,:,:,1,1));  % dr/dx
Jry = tensor3(JM,1,JM,Jac.*Rx(:,:,:,1,2));  % dr/dy
Jsx = tensor3(JM,1,JM,Jac.*Rx(:,:,:,2,1));  % ds/dx
Jsy = tensor3(JM,1,JM,Jac.*Rx(:,:,:,2,2));  % ds/dy


% Interpolate the velocity onto quadrature points
Cx = tensor3(JM,1,JM,Cx);
Cy = tensor3(JM,1,JM,Cy);

% Collocate on fine mesh 
Cr = BMh.*(Jrx.*Cx + Jry.*Cy);
Cs = BMh.*(Jsx.*Cx + Jsy.*Cy);

