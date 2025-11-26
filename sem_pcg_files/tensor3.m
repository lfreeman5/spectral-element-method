function F = tensor3(Az,Ay,Ax,F);
hdr

%
%  Note, this code also works if Ax, Ay, or Az are constants instead of matrices
%

[nx,ny,nz]=size(F);

[mx,nxa]=size(Ax);
[my,nya]=size(Ay);
[mz,nza]=size(Az);

if mx==1 && nxa==1; mx=nx; end;    %% Handle the 1x1 Ax case
if my==1 && nya==1; my=ny; end;    %% Handle the 1x1 Ay case
if mz==1 && nza==1; mz=nz; end;    %% Handle the 1x1 Az case

if mx*nxa > 1;  
   F=reshape(F,nx,ny*nz); F=Ax*F; 
else 
   if Ax ~= 1;  F=Ax*F; end;
end;

if mz*nza > 1;  
   F=reshape(F,mx*ny,nz); F=F*Az'; 
else 
   if Az ~= 1;  F=Az*F; end;
end;

if my*nya > 1; F=reshape(F,mx,ny,mz); 
  if ny==my; for k=1:mz; F(:,:,k) = F(:,:,k)*Ay'; end;
  else; 
    Fy=zeros(mx,my,mz);
    for k=1:mz; Fy(:,:,k) = F(:,:,k)*Ay'; end; F=Fy;
  end;
else;
  if Ay ~= 1; F=Ay*F; end;
  F=reshape(F,mx,my,mz);
end;

