function F = tensor4(A4,Az,Ay,Ax,F);

%
%  Works also if Ai are constants instead of matrices
%

[nx,ny,nz,n4]=size(F);

[mx,nxa]=size(Ax);
[my,nya]=size(Ay);
[mz,nza]=size(Az);
[m4,n4a]=size(A4);

if mx==1 && nxa==1; mx=nx; end;    %% Handle the 1x1 Ax case
if my==1 && nya==1; my=ny; end;    %% Handle the 1x1 Ay case
if mz==1 && nza==1; mz=nz; end;    %% Handle the 1x1 Az case
if m4==1 && n4a==1; m4=n4; end;    %% Handle the 1x1 A4 case

if Ax ~= 1; F=reshape(F,nx,ny*nz*n4); F=Ax*F;  end;
if A4 ~= 1; F=reshape(F,mx*ny*nz,n4); F=F*A4'; end;

if Ay ~= 1; F=reshape(F,mx,ny,nz*m4); 
  if ny==my; for k=1:nz; F(:,:,k) = F(:,:,k)*Ay'; end;
  else; 
    Ft=zeros(mx,my,nz*m4);
    for k=1:nz*m4; Ft(:,:,k) = F(:,:,k)*Ay'; end; F=Ft;
  end;
end;

if Az == 1; F=reshape(F,mx,my,mz,m4); end;

if Az ~= 1; F=reshape(F,mx*my,nz,m4); 
  if nz==mz; for k=1:m4; F(:,:,k) = F(:,:,k)*Az'; end;
  else; 
    Ft=zeros(mx*my,nz,m4);
    for k=1:m4; Ft(:,:,k) = F(:,:,k)*Az'; end; F=Ft;
  end;
  F=reshape(F,mx,my,mz,m4);
end;

