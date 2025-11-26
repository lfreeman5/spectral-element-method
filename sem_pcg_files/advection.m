hdr;    % 2-D SEM multi-element
%close all;

Nelx = 13;  Nely = 2; E = Nelx*Nely;
N=10; N1=N+1; 


%% Base Geometry
x0 =  -3.0;  x1=1.0;  Lx=x1-x0;   % Domain coordinates and size
y0 =  -1.0;  y1=1.0;  Ly=y1-y0;

%% Circular Geometry
x0 =  -1.5;  x1=-x0;  Lx=x1-x0;   % Domain coordinates and size
y0 =   1.0;  y1=1.5;  Ly=y1-y0;

zc = zwuni(Nelx); xc = x0 + Lx*(zc+1)/2;
zc = zwuni(Nely); yc = y0 + Ly*(zc+1)/2;

%% Problem parameters, as function of N

[z,w,Dh,X,Y,Grr,Grs,Gss,Bl,Xr,Rx,Jac,Q,glo_num,M,dA]=set_sem(N,Nelx,Nely,xc,yc);

Nd = floor(1.5*N);
[z,w]=zwgll(N);
[zd,wd]=zwgl(Nd); Bd=diag(wd);
[zd,wd]=zwgll(N); Bd=diag(wd);
JM=interp_mat(zd,z);
DM=deriv_mat (zd);
BMh=tensor3(Bd*JM,1,Bd*JM,1+0*Jac); %% effectively, Bd*JM*Jac*JM'*Bd'

Nf = floor(1.2*N); [zf,wf]=zwuni(Nf); Jf=interp_mat(zf,z);

U = -1+0*X;  %% PLANE TRANSLATION
V =  0+0*X;
U = -Y;  %% PLANE ROTATION
V =  X;

[Cr,Cs]=set_advect_c(U,V,JM,BMh,Jac,Rx);

[lam_max_est]=est_lam_cfl(U,V,Rx);
ldt_max = 0.5;
dt=ldt_max/lam_max_est;

Tfinal = 4*pi;
nsteps = ceil(Tfinal/dt);
dt     = Tfinal/nsteps;

XY2 = (X-.5).^2+Y.^2;
T =  exp(-50*XY2);
T =  exp(-10*X.*X).*exp(.3*Y);
T = l2proj(T,Q,M,Bl);
T0=T;

T1=0*T; T2=0*T; T3=0*T;
H1=0*T; H2=0*T; H3=0*T;


b0 = 0; nu=1.e-20; alpha=1.e-2;


%%%%% TIME STEPPING LOOP %%%%%%

kk=0; k=0;
for iloop=1:1;
  for istep =1:nsteps; k=k+1;

%    [Cr,Cs,lam_max_est]=set_advect_c(U,V,JM,BMh,Jac,Rx);
%    dt=0.4/lam_max_est; 
     time = k*dt;

     ndt = nu*dt;
     adt = alpha*dt;
     if k==1; a1=1; a2=0; a3=0; b0=1; b1=1; b2=0; b3=0; end;
     if k==2; a1=1.5; a2=-.5; a3=0; b0=1.5; b1=2; b2=-.5; b3=0; end;
     if k==3; a1=3; a2=-3; a3=1; b0=11/6; b1=3; b2=-1.5; b3=2/6; end;
     d1=dt*a1; d2=dt*a2; d3=dt*a3;


%%   Compute t-hat

     T3=T2;T2=T1;T1=T; Ql=0*T;
        H3=H2;H2=H1;H1=-advectl(T,Cr,Cs,JM,DM) + Bl.*Ql;
        Th=Bl.*(b1*T1+b2*T2+b3*T3)+(d1*H1+d2*H2+d3*H3);
%       Th=Th - adt*( ATBx*T*BTBy' + BTBx*T*ATBy' ); %% T=0 on Boundary for now
%       Th=Th - b0*( BTBx*T*BTBy'); % Dirichlet

%%   Implicit solve

     ifnull=0;
     tol=1.e-9;
     max_iter=140;

     dAT=1./dA;
     dAT=b0*qqt(Q,Bl)+adt*dAT;
     dAT=1./dAT;

     [T,iter,res,lamda_h]=...
      pcg_lambda(Th,tol,max_iter,b0,adt,M,Q,Bl,Grr,Grs,Gss,Dh,dAT,ifnull);

%    Diagonostics

     if mod(istep,10)  ==0 || istep==1;  kk=kk+1;

       tmax = max(max(max(abs(T))));

       hold off;
       tm(kk) = tmax; ti(kk) = time;

       Xf = tensor3(Jf,1,Jf,X);
       Yf = tensor3(Jf,1,Jf,Y);
       Tf = tensor3(Jf,1,Jf,T);

%      if istep>2; Tf=Tf-Tfl; end;
%      Tfl = tensor3(Jf,1,Jf,T);
%      tmax = max(max(max(abs(Tf))));

       s=['Time,UVT_{max}: ' num2str(time) ',   ' num2str(tmax) ,...
          ', ' num2str(istep) ', ' num2str(iter)'.'];
       se_mesh (Xf,Yf,Tf,s);  axis equal; hold on; 
       drawnow

     end;

     if tmax > 1.9; break; end;

   end;
end;

