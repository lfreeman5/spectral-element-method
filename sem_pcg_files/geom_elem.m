function [Grr,Grs,Gss,B,Xr,Rx,Jac]=geom_elem(X,Y,Dh,w);

hdr;

N1= length(w);  N=N1-1;
E = size(X,2);

Grr = 0*X; Grs=Grr; Gss=Grr; Jac=Grr; B=Gss;

Be=w*w';
for e=1:E
    B(:,e,:)=Be(:,:);
end;


Xr=zeros(N1,E,N1,2,2);
Rx=zeros(N1,E,N1,2,2);

%  Compute dx_i / dr_j

Xr(:,:,:,1,1) = tensor3(1,1,Dh,X);              % dx/dr: 
Xr(:,:,:,1,2) = tensor3(Dh,1,1,X);              % dx/ds: 
Xr(:,:,:,2,1) = tensor3(1,1,Dh,Y);              % dx/dr: 
Xr(:,:,:,2,2) = tensor3(Dh,1,1,Y);              % dx/ds: 

Jac = Xr(:,:,:,1,1).*Xr(:,:,:,2,2) - Xr(:,:,:,1,2).*Xr(:,:,:,2,1); 
Ji  = 1./Jac;

Rx(:,:,:,1,1) =  Ji.*Xr(:,:,:,2,2);                       % dr/dx
Rx(:,:,:,1,2) = -Ji.*Xr(:,:,:,1,2);                       % dr/dy
Rx(:,:,:,2,1) = -Ji.*Xr(:,:,:,2,1);                       % ds/dx
Rx(:,:,:,2,2) =  Ji.*Xr(:,:,:,1,1);                       % ds/dy

B = Jac.*B; % nr x ns diagonal mass matrix on Omega-hat, incl. Jac

Grr=B.*(Rx(:,:,:,1,1).*Rx(:,:,:,1,1)+Rx(:,:,:,1,2).*Rx(:,:,:,1,2));% G_rr=sum_j dr/dx_j * dr/dx_j
Grs=B.*(Rx(:,:,:,1,1).*Rx(:,:,:,2,1)+Rx(:,:,:,1,2).*Rx(:,:,:,2,2));% G_rs=sum_j dr/dx_j * ds/dx_j
Gss=B.*(Rx(:,:,:,2,1).*Rx(:,:,:,2,1)+Rx(:,:,:,2,2).*Rx(:,:,:,2,2));% G_ss=sum_j ds/dx_j * ds/dx_j 

