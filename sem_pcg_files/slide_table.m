hdr;    % 2-D SEM multi-element
close all;
format shorte;

Nelx = 15; Nely =  5; E = Nelx*Nely;

N=10; N1=N+1; 

%% Base Geometry
x0 =  -1.5;  x1=-x0;  Lx=x1-x0;   % Domain coordinates and size
y0 =   1.0;  y1=1.5;  Ly=y1-y0;

zc = zwuni(Nelx); xc = x0 + Lx*(zc+1)/2;
zc = zwuni(Nely); yc = y0 + Ly*(zc+1)/2;

%% Problem parameters, as function of N

[z,w,Dh,X,Y,Grr,Grs,Gss,Bl,Xr,Rx,Jac,Q,glo_num,M,dA]=set_sem(N,Nelx,Nely,xc,yc);
E2 = ceil(E/2); M(:,E2,:)=0; M=qqt_op(M,'*',glo_num);
Bb = qqt(Q,Bl);
Bi = 1./Bb;  %% Inverse of diagonal-assembled mass matrix
vol=sum(sum(sum(Bl)));

%%% SET Two-Level MG VARIABLES
Nc=4;   % Coarse-to-fine interpolator for multigrid
[zc,wc]=zwgll(Nc); Jc=interp_mat(z,zc); Dc=deriv_mat(zc);
[Grrc,Grsc,Gssc,Blc,Qc,glo_numc,Mc,dAc]=set_crs(Nc,Nelx,Nely,xc,yc);
Mc(:,E2,:)=0; Mc=qqt_op(Mc,'*',glo_numc);

table_hdr = ...
   [' ';'   N   Nc    k       e_inf(k)      smoother         tol_c    kc'];

for ipass=1:2;

  disp(table_hdr)

  if ipass==1; 
    Ue = rand(N1,E,N1);
    Ue = Bi.*qqt(Q,M.*Bl.*Ue);             %% Project Ue into H1_0
    Fl = axl(Ue,b0,nu,Bl,Grr,Grs,Gss,Dh);
    se_mesh(X,Y,Ue,'Solution')
  end;

  b0 = 0; nu=1; ifnull=0;                  %% Define Poisson
  [Ul,iter,res,lam_max]=...                %% Get lam_max estimate
      pcg_lambda(Fl,tol,max_iter,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,dA,ifnull);
  lam_bnd=1.1*lam_max;                     %% Lambda upper bound for Chebyshev iteration

  rl=Fl; Umg=0*Fl;
  for mg_iter=1:10;

    %% Smoothing step

    if ipass==1;
       omega=0.666666;
       [Us]=smoother(rl,lam_bnd,omega,5,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,dA,ifnull);
    else;
       [Us]=cheby4  (rl,lam_bnd,omega,6,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,dA,ifnull);
    end;
    Umg = Umg+Us;

    err = Ue-Umg; ems = max(max(max(abs(err))));  %% For diagnostics only


    %% Coarse-grid correction

    rl = rl-axl(Us,b0,nu,Bl,Grr,Grs,Gss,Dh);      %% Update residual
    rc = tensor3(Jc',1,Jc',rl);
    nrc= sqrt( sum(sum(sum(M.*dA.*(rl.^2))))/vol );
    tolc=0.1*nrc;
    [Ec,iterc,res,lam_crs]=...
        pcg_lambda(rc,tolc,100,b0,nu,Mc,Qc,Blc,Grrc,Grsc,Gssc,Dc,dAc,ifnull);
    El = tensor3(Jc,1,Jc,Ec);

    Umg = Umg + El;                            %% Add coarse-grid correction
    rl  = rl-axl(El,b0,nu,Bl,Grr,Grs,Gss,Dh);


    err = Ue-Umg; emx = max(max(max(abs(err))));  %% For diagnostics only
    caption = ['Two-Level Error, e_{inf} = ' num2str(emx)];
    figure; se_mesh(X,Y,err,caption);
%   figure; se_mesh(X,Y,qqt(Q,M.*rl),'Residual');

    s=sprintf('%4d %4d %4d %13.5f  %13.5f  %13.5f %4d',[N Nc mg_iter emx ems nrc iterc]);
    disp(s)

end;

  end;
