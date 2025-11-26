hdr;    % 2-D SEM multi-element



%% Geometry
x0 =  0.0;  x1=1.0; Lx=x1-x0;   % Domain coordinates and size
y0 =  0.0;  y1=1.0; Ly=y1-y0;


Nelx = 5; 
Nely = 5; 

N=08; Nx=N; Ny=N;

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
vol = sum(sum(sum(B)))

[Q,glo_num]=set_tp_semq(Nelx,Nely,N);

% gsum = qqt(Q,glo_num);
% se_disp(gsum,'gsum')

se_mesh(X,Y,glo_num,'glo_num')

M=set_mask('N','N','D','D',N,Nelx,Nely); Ue=sin(pi*Y);
M=set_mask('D','D','D','D',N,Nelx,Nely); Ue=sin(pi*X).*sin(2*pi*Y);
M=set_mask('D','D','D','D',N,Nelx,Nely); Ue=sin(pi*X).*sin(pi*Y);

se_mesh(X,Y,Ue,'Exact')

b0 = 0; nu=1;
b0 = 0; nu=1+0*X; nu(:, 7:9,:)=100;
                  nu(:,12:14,:)=100;
                  nu(:,17:19,:)=100;
Fl = axl(Ue,b0,nu,Bl,Grr,Grs,Gss,Dh);
Fl = (1+5*pi*pi)*Bl.*Ue;
Fl = Bl;

ifnull=0;
tol=1.e-8;
max_iter=400;
[U,iter,res]=pcg_sem(Fl,tol,max_iter,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,ifnull);

se_mesh(X,Y,U,'Solution')


