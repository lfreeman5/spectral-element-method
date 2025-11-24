hdr;    % 2-D SEM multi-element
close all;

Nelx = 15; Nely =  5; E = Nelx*Nely;

N=08; N1=N+1; 

%% Base Geometry
x0 =  -1.5;  x1=-x0;  Lx=x1-x0;   % Domain coordinates and size
y0 =   1.0;  y1=1.5;  Ly=y1-y0;

zc = zwuni(Nelx); xc = x0 + Lx*(zc+1)/2;
zc = zwuni(Nely); yc = y0 + Ly*(zc+1)/2;

%% Problem parameters, as function of N

[z,w,Dh,X,Y,Grr,Grs,Gss,Bl,Xr,Rx,Jac,Q,glo_num,M,dA]=set_sem(N,Nelx,Nely,xc,yc);
E2 = ceil(E/2); M(:,E2,:)=0; M=qqt_op(M,'*',glo_num);

b0 = 0; nu=1;

Ue = rand(N1,E,N1);
Fl = axl(Ue,b0,nu,Bl,Grr,Grs,Gss,Dh);

ifnull=0; tol=1.e-8; max_iter=400;
[Ul,iter,res,lam_max]=...
    pcg_lambda(Fl,tol,max_iter,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,dA,ifnull);
lam_max=1.1*lam_max;

%%% SET Two-Level MG VARIABLES

Nc=N-2; Nc=max(1,Nc); 
[zc,wc]=zwgll(Nc); Dc=deriv_mat(zc); Jc=interp_mat(z,zc); %C-to-F interpolator 

[Grrc,Grsc,Gssc,Blc,Qc,glo_numc,Mc,dAc]=set_crs(Nc,Nelx,Nely,xc,yc);
Mc(:,E2,:)=0; Mc=qqt_op(Mc,'*',glo_numc);

omega=0; tol=1.e-8; max_iter=10;
[Ul,res,iter]=mg_ksp(Fl,lam_max,omega,tol,max_iter,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,dA,ifnull,...
                        Jc,Mc,Qc,Blc,Grrc,Grsc,Gssc,Dc,dAc);

caption = ['Smoothed Error, e_{inf} = ' num2str([iter res])];
figure; se_mesh(X,Y,err,caption);

