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

% Ue = rand(N1,E,N1);
% Fl = axl(Ue,b0,nu,Bl,Grr,Grs,Gss,Dh);

ifnull=0;
tol=1.e-8;
max_iter=400;
[U,iter,res]=pcg_sem(Fl,tol,max_iter,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,ifnull);
% se_mesh(X,Y,U,'Solution')

[Ul,iter,res,lam_max]=...
    pcg_lambda(Fl,tol,max_iter,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,dA,ifnull);
se_mesh(X,Y,Ul,'Solution')
lam_max=1.1*lam_max;

%%% SET Two-Level MG VARIABLES

Nc=N; 
Nc=ceil(1+Nc/2); Nc=max(1,Nc); [zc,wc]=zwgll(Nc);
Jc=interp_mat(z,zc);   % Coarse-to-fine interpolator for multigrid
Dc=deriv_mat(zc);

[Grrc,Grsc,Gssc,Blc,Qc,glo_numc,Mc,dAc]=set_crs(Nc,Nelx,Nely,xc,yc);
Mc(:,E2,:)=0; Mc=qqt_op(Mc,'*',glo_numc);

rl=Fl; Umg=0*Fl;


for mg=1:10;

  %% Smoothing step
  omega=0.666666;
% [Us]=smoother(rl,lam_max,omega,5,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,dA,ifnull);
  [Us]=cheby4(rl,lam_max,omega,6,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,dA,ifnull);
  Umg = Umg+Us;

  err = Ul-Umg; ems = max(max(max(abs(err))));
% caption = ['Smoothed Error, e_{inf} = ' num2str(ems)];
% figure; se_mesh(X,Y,err,caption);


  %% Coarse-grid correction
  rl = Fl-axl(Umg,b0,nu,Bl,Grr,Grs,Gss,Dh);
  rc = tensor3(Jc',1,Jc',rl);
  tolc=0.1*ems;
  [Ec,iterc,res,lam_crs]=...
      pcg_lambda(rc,tolc,100,b0,nu,Mc,Qc,Blc,Grrc,Grsc,Gssc,Dc,dAc,ifnull);
  El = tensor3(Jc,1,Jc,Ec);

  Umg = Umg + El;

  err = Ul-Umg; emx = max(max(max(abs(err))));
  caption = ['Two-Level Error, e_{inf} = ' num2str(emx)];
  figure; se_mesh(X,Y,err,caption);

  format shorte;
  disp([mg ems emx iterc])

  rl = Fl-axl(Umg,b0,nu,Bl,Grr,Grs,Gss,Dh);

end;
