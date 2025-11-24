hdr;

N=20;

Lx=2*pi;       % Domain size
Ly=2; 

nu=0.1;
alpha=0.1;


tfinal = 50;
CFL    = 0.25;


[Ah,Bh,Ch,Dh,z,w] = semhat(N);
nh = N+1; Ih = speye(nh); 

M=ceil(5+1.2*N); [zm,wm]=zwgll(M); Jh=interp_mat(zm,z); Dt=Jh*Dh;
BMxy = (Lx*Ly/4)*wm*wm';



Lx2=Lx/2; Lxi=1./Lx2;
Ly2=Ly/2; Lyi=1./Ly2;

x=Lx*(z+1)/2; y=Ly*z/2; [X,Y]=ndgrid(x,y);  

Nf = 5+ceil(1.7*N); [zf,wf]=zwuni(Nf); Jf=interp_mat(zf,z);   % For plotting
Xf=Jf*X*Jf'; Yf=Jf*Y*Jf';           % For plotting
Xm=min(min(X)); Ym=min(min(Y));     % For plotting
XM=max(max(X)); YM=max(max(Y));     % For plotting

O=0*X; U=O; V=O; T=O;  F=O; G=O; H=O;
U1=O;U2=O;U3=O; V1=O;V2=O;V3=O;
F1=O;F2=O;F3=O; G1=O;G2=O;G3=O;
f1=O;f2=O;f3=O; g1=O;g2=O;g3=O;
T1=O;T2=O;T3=O; H1=O;H2=O;H3=O;


Bhh = (w*w');
Bxy = (Lx*Ly/4)*Bhh;

RUx=Ih(1:end-1,:); RUx(1,end)=1;  %% Periodic in X
RVx=RUx;                          %% Periodic in X
RPx=RUx;                          %% Periodic in X
RTx=RUx;                          %% Periodic in X

RUy=Ih(2:end-1,:);                %% Dirichlet velocity in Y
RVy=Ih(2:end-1,:);                %% Dirichlet velocity in Y
RPy=Ih;                           %% Neumann pressure in Y
RTy=Ih(2:end-1,:);                %% Dirichlet temperature in Y

% Generate Eigenpairs: 
%    S=Matrix of eigenvectors
%    D=Vector of eigenvalues

AUx = Lxi*RUx*Ah*RUx'; BXx = Lx2*RUx*Bh*RUx'; [SUx,DUx]=gen_eig_ortho(AUx,BXx);
AVx = Lxi*RVx*Ah*RVx'; BYx = Lx2*RVx*Bh*RVx'; [SVx,DVx]=gen_eig_ortho(AVx,BYx);
ATx = Lxi*RTx*Ah*RTx'; BTx = Lx2*RTx*Bh*RTx'; [STx,DTx]=gen_eig_ortho(ATx,BTx);
APx = Lxi*RPx*Ah*RPx'; BPx = Lx2*RPx*Bh*RPx'; [SPx,DPx]=gen_eig_ortho(APx,BPx);
AUy = Lyi*RUy*Ah*RUy'; BXy = Ly2*RUy*Bh*RUy'; [SUy,DUy]=gen_eig_ortho(AUy,BXy);
AVy = Lyi*RVy*Ah*RVy'; BYy = Ly2*RVy*Bh*RVy'; [SVy,DVy]=gen_eig_ortho(AVy,BYy);
ATy = Lyi*RTy*Ah*RTy'; BTy = Ly2*RTy*Bh*RTy'; [STy,DTy]=gen_eig_ortho(ATy,BTy);
APy = Lyi*RPy*Ah*RPy'; BPy = Ly2*RPy*Bh*RPy'; [SPy,DPy]=gen_eig_ortho(APy,BPy);

% Prolongate reconstruction (eigen) vectors

SUx = RUx'*SUx; SUy = RUy'*SUy;   
SVx = RVx'*SVx; SVy = RVy'*SVy;
STx = RTx'*STx; STy = RTy'*STy;
SPx = RPx'*SPx; SPy = RPy'*SPy;

% Set up pointwise entries in (k,l) location, lambda_kl = lambda_k + lambda_l

ex=1+0*DUx; ey=1+0*DUy; XLam=ex*DUy' + DUx*ey';
ex=1+0*DVx; ey=1+0*DVy; YLam=ex*DVy' + DVx*ey';
ex=1+0*DTx; ey=1+0*DTy; TLam=ex*DTy' + DTx*ey';
ex=1+0*DPx; ey=1+0*DPy; PLam=ex*DPy' + DPx*ey';

PLam(1,1) = 1;  DPi=1./PLam; DPi(1,1) = 0;  %% Neumann operator for pressure

dx   = min(diff(x)); dy   = min(diff(y)); dx=min(dx,dy);
c    = 2;
if c>0;  dt = CFL*(dx/c);   end; 
if c==0; dt = CFL*dx;       end; 
nsteps = floor(tfinal/dt);
iostep = floor(nsteps/90);
nsteps = iostep*ceil(nsteps/iostep);
dt     = tfinal/nsteps;

U=0*X;
V=0*Y;

tstart=tic; k=0; kk=0; clear tt uu;
for iloop=1:1;
  for istep =1:nsteps; k=k+1; time = k*dt;

     ndt = nu*dt;
     adt = alpha*dt;
     if k==1; a1=1; a2=0; a3=0; b0=1; b1=1; b2=0; b3=0; end;
     if k==2; a1=1.5; a2=-.5; a3=0; b0=1.5; b1=2; b2=-.5; b3=0; end;
     if k==3; a1=3; a2=-3; a3=1; b0=11/6; b1=3; b2=-1.5; b3=2/6; end;
     if k<4; DUi=1./(ndt*XLam+b0); DVi=1./(ndt*YLam+b0); DTi=1./(adt*TLam+b0); end;
     d1=dt*a1; d2=dt*a2; d3=dt*a3;


     Ur = Lxi*BMxy.*(Jh*U*Jh'); Vr = Lyi*BMxy.*(Jh*V*Jh'); 

     if k==1; Q  =  0 + .0*Y; end;
     if k==1; FX =  1 + .0*Y; end;
     if k==1; FY =  0 + .0*Y; end;

     Omega = Lxi*(Dh*V) - Lyi*(U*Dh');      %% Vorticity
     curlcurlX =  Bxy.*(Lyi*(Omega*Dh'));
     curlcurlY = -Bxy.*(Lxi*(Dh*Omega));

     U3=U2;U2=U1;U1=U;
        F3=F2;F2=F1;F1=-advect(U,Ur,Vr,Jh,Dt)+Bxy.*FX;
        f3=f2;f2=f1;f1=-nu*curlcurlX;
        Uh=Bxy.*(b1*U1+b2*U2+b3*U3)+(d1*F1+d2*F2+d3*F3);
        Ut=Uh+(d1*f1+d2*f2+d3*f3);

     V3=V2;V2=V1;V1=V;
        G3=G2;G2=G1;G1=-advect(V,Ur,Vr,Jh,Dt)+Bxy.*FY;
        g3=g2;g2=g1;g1=-nu*curlcurlY;
        Vh=Bxy.*(b1*V1+b2*V2+b3*V3)+(d1*G1+d2*G2+d3*G3);
        Vt=Vh+(d1*g1+d2*g2+d3*g3);

     T3=T2;T2=T1;T1=T;H3=H2;H2=H1;H1=-advect(T,Ur,Vr,Jh,Dt)+Bxy.*Q;
        Th=Bxy.*(b1*T1+b2*T2+b3*T3)+(d1*H1+d2*H2+d3*H3);

%    pressure correction
     divUT = Lxi*Dh'*(Ut) + Lyi*(Vt)*Dh;
     P = (1./dt)*( SPx*(DPi.*(SPx'*divUT*SPy))*SPy');
     dPdx = Lxi*(Dh*P); dPdy = Lyi*(P*Dh');

     Uh = Uh - dt*Bxy.*dPdx;
     Vh = Vh - dt*Bxy.*dPdy;

%    viscous solve
     U = SUx*(DUi.*(SUx'*Uh*SUy))*SUy';  %% Already prolongated to boundary
     V = SVx*(DVi.*(SVx'*Vh*SVy))*SVy';
     T = STx*(DTi.*(STx'*Th*STy))*STy';

%    Diagonostics
     umax=max(max(abs(U)));
     vmax=max(max(abs(V)));
     tmax=max(max(abs(T)));

     exact = (1/(2*nu))*(1-Y.*Y);
     error = exact-U;
     emax  = max(max(abs(error)));

     emx(k) = emax;
     umx(k) = umax;
     ttm(k) = time;

     if umax > 10; break; end;

     if mod(k,500)==0 || k==1;  kk=kk+1;
       uu(kk) = vmax; tt(kk) = time;
       s=['Time,UVT_{max}: ' num2str(time) ',   ' num2str(umax) ',   ' ...
                       num2str(vmax) ',   ' num2str(tmax) '.'];
       sc = 1./umax;
       T = U;
       Uf=Jf*U*Jf'; Vf=Jf*V*Jf'; Tf=Jf*(T)*Jf';
       hold off; quiver  (X,Y,U,V,'k-'); axis equal; 
       axis([Xm XM Ym YM ]); axis off;
       title(s,fs,12); drawnow
     end;

   end;
end;
elapsed_time = toc(tstart);

figure
model = min(5.0,ttm);
lam1  = nu*pi*pi/4;
moder = 5*exp(-lam1*ttm);
plot(ttm,umx,'k-',ttm,model,'r--',...
     ttm,emx,'b-',ttm,moder,'r--',lw,2);
xlabel('time, t',fs,20);
ylabel('U_{max}',fs,20);
title('History for Plane Poiseiulle Flow',fs,16);

figure
model = min(5.0,ttm);
lam1  = nu*pi*pi/4;
moder = 5*exp(-lam1*ttm);
scale = 32/(pi.^3);
mode2 = scale*5*exp(-lam1*ttm);
semilogy(ttm,emx,'b-',ttm,moder,'r--',lw,2);
xlabel('time, t',fs,20);
ylabel('Error',fs,20);
title('History for Plane Poiseiulle Flow',fs,16);

figure
model = min(5.0,ttm);
lam1  = nu*pi*pi/4;
moder = 5*exp(-lam1*ttm);
scale = 32/(pi.^3);
mode2 = scale*5*exp(-lam1*ttm);
semilogy(ttm,abs(emx-moder),'b-',lw,2,ttm,abs(emx-mode2),'r--',lw,2);
xlabel('time, t',fs,20);
ylabel('Error',fs,20);
title('History for Plane Poiseiulle Flow',fs,16);

