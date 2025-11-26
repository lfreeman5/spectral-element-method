
%  2D Advection on Omega = [-1,1]

%  Failure mode:  Inflow coming in outflow BC

hdr; hold off;

if_rk4 = 0;

disp(' ')

N=40; nu = .001;

[Ah,Bh,Ch,Dh,z,w] =  semhat(N); Ih=speye(N+1);
x=z; Lx = max(x)-min(x);  Lx2=Lx/2; Lxi=2/Lx;
y=z; Ly = max(y)-min(y);  Ly2=Ly/2; Lyi=2/Ly;

[X,Y] = ndgrid(x,y);    %% Domain

Cx = -X*0+1;            %% Advecting Velocity Field C=(Cx,Cy)
Cy =  Y*0+0;   

U0 = 0*X;               %% Homogeneous IC

delta = 0.5;            %% Gaussian Pulse for Suurce
X0 = -.5; Y0 = 0.0;  
x = X-X0; y = Y-Y0; arg = -(x.*x+y.*y)/(delta^2);
Q0 = exp(arg);


Rx = Ih(2:end-0,:);     %% Restriction matrics for BCs
Ry = Ih(2:end-1,:);
                                                                      % Prolongate S (faster)
Ax = Lxi*Rx*Ah*Rx'; Bx = Lx2*Rx*Bh*Rx'; [Sx,Dx]=gen_eig_ortho(Ax,Bx);   Sx = Rx'*Sx; 
Ay = Lyi*Ry*Ah*Ry'; By = Ly2*Ry*Bh*Ry'; [Sy,Dy]=gen_eig_ortho(Ay,By);   Sy = Ry'*Sy;
Bb = (Lx2*Ly2)*w*w';

ex=1+0*Dx; ey=1+0*Dy; Lam=ex*Dy'+Dx*ey'; %% Diagonal for Poisson Operator

%%% The Next Few Lines Are for Checking the CFL

[z1,w1] = zwgll(N-1);
J1 = interp_mat(z1,z); % interpolate to GLL "midpoints" for CFL check

dX = diff(X);  dX = dX*J1';         Cxm = J1*Cx*J1'; dUdx = Cxm./dX;
dY = diff(Y'); dY = dY'; dY=J1*dY;  Cym = J1*Cy*J1'; dUdy = Cym./dY;

dUda = abs(dUdx)+abs(dUdy);
dUdm = max(max(dUda))
X1 = J1*X*J1';
Y1 = J1*Y*J1';

CFL    = 0.60000                    %% This is for BDF3/EXT3
if if_rk4 > 0; CFL = 2.82685, end;  %% This is for RK4
dtmax  = CFL/dUdm
tau    = 6
mstep  = tau/dtmax
nstep  = 2000
dt     = tau/nstep
CFL    = dt*dUdm

nlaps  = 0.4            % Number of laps = nlaps
nsteps = nlaps*nstep



U = U0; dt2 = dt/2; dt6=dt/6;
emax = 0;

U1=0*U0; U2=U1; U3=U1; F1=U1; F2=U1; F3=U1;  % Zero-out lagged variables at time t=0

for istep=1:nsteps; time = dt*istep;

    if if_rk4 > 0 ;
       k1 = -Mask.*( Cx.*(Dh*U ) + Cy.*(U *Dh')+W*(H*U*H')); U1 = U + dt2*k1;
       k2 = -Mask.*( Cx.*(Dh*U1) + Cy.*(U1*Dh')+W*(H*U*H')); U2 = U + dt2*k2;
       k3 = -Mask.*( Cx.*(Dh*U2) + Cy.*(U2*Dh')+W*(H*U*H')); U3 = U + dt *k3;
       k4 = -Mask.*( Cx.*(Dh*U3) + Cy.*(U3*Dh')+W*(H*U*H')); 
       U  = U + dt6*(k1 + 2*(k2 + k3) + k4);
    else;
%%     Set updated BDFk/EXTk coefficients
       ndt = nu*dt; 
       if istep==1; a1=1; a2=0; a3=0; b0=1; b1=1; b2=0; b3=0; end;
       if istep==2; a1=1.5; a2=-.5; a3=0; b0=1.5; b1=2; b2=-.5; b3=0; end;
       if istep==3; a1=3; a2=-3; a3=1; b0=11/6; b1=3; b2=-1.5; b3=2/6; end;
       d1=dt*a1; d2=dt*a2; d3=dt*a3;
       if istep<4; Di=1./(ndt*Lam+b0); end;   % Set diagonal for Fast Poisson Solver

       U3=U2;U2=U1;U1=U;                                        % BDFk terms
       F3=F2;F2=F1;F1=-Bb.*( Cx.*(Dh*U ) + Cy.*(U *Dh') - Q0);  % EXTk terms
       Uh=Bb.*(b1*U1+b2*U2+b3*U3)+(d1*F1+d2*F2+d3*F3);          % Gather RHS

       U = Sx*(Di.*(Sx'*Uh*Sy))*Sy';         %  Viscous Solve - prolongated to boundary

    end;

    umax = max(max(abs(U)));
%   if umax > 10; break; end;
%   if umax < 100*eps; disp('Signal Gone.'), break; end;

    if mod(istep,1)==0;
       X0t =  X0 + time*Cx;
       Y0t =  Y0;
       x = X-X0t; y = Y-Y0t; arg = -(x.*x+y.*y)/(delta^2);
       Ue = exp(arg);
       Er = Ue-U;
       lmax = max(max(abs(Er)));
       emax = max(emax,lmax);
    end;

    if mod(istep,5)==0;
       Ulog = log10(abs(U)+eps);
%      mesh(X,Y,Ulog);  zlabel('log U',fs,20);
       mesh(X,Y,U);  zlabel('U',fs,20);
%      mesh(X,Y,Er); zlabel('Error',fs,20);
       xlabel('X',fs,20); ylabel('Y',fs,20);
       title(['Time = ' num2str([time istep nsteps dt N umax])],fs,15);
%      axis([-1 1 -1 1 0 1]);
       drawnow;
    end;

end;

jtrial=jtrial+1;

jN (jtrial)=N;
jdt(jtrial)=dt;
jem(jtrial)=emax;
jtm(jtrial)=time;
jsm(jtrial)=mstep;
jsn(jtrial)=nstep;
jcf(jtrial)=CFL;

nt=length(jN); it=[1:nt];
disp([it' jN' jdt' jem' jtm' 2*pi./jdt' jsm' jsn' jcf'])

