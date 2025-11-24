hdr;    % 2-D SEM multi-element


%% Geometry
x0 =  0.0;  x1=1.0; Lx=x1-x0;   % Domain coordinates and size
y0 =  0.0;  y1=1.0; Ly=y1-y0;


Nelx = 1; 
Nely = 2; 

N=02; Nx=N; Ny=N;

[z,w]=zwgll(N); N1=N+1; [R,S]=ndgrid(z,z);

ifourier=0;
[Ah,Bh,Ch,Dh,Ih,J,z,w,Jf,zf]= lfsemhat(N,ifourier);  % Set basic operators

% Mesh values

zc=zwuni(Nelx); xc = x0 + Lx*(zc+1)/2;
zc=zwuni(Nely); yc = y0 + Ly*(zc+1)/2;
E = Nelx*Nely;

X=zeros(N1,E,N1); Y=X;

e=0; 
for ey=1:Nely; for ex=1:Nelx; e=e+1;
    xe0=xc(ex); xe1=xc(ex+1); xeL=xe1-xe0;
    ye0=yc(ey); ye1=yc(ey+1); yeL=ye1-ye0;
    X(:,e,:) = xe0 + xeL*(R+1)/2;
    Y(:,e,:) = ye0 + yeL*(S+1)/2;
end; end;

[Grr,Grs,Gss,Bl,Xr,Rx,Jac]=geom_elem(X,Y,Dh,w);
vol = sum(sum(sum(Bl)))

[Q,glo_num]=set_tp_semq(Nelx,Nely,N);

se_disp(glo_num,'glo_num')

gsum = qqt(Q,glo_num);
se_disp(gsum,'gsum')

