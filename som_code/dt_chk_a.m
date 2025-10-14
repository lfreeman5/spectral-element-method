
%  2D Advection on Omega = [-1,1]

hdr; hold off;


disp(' ')

% N=80;

[Ah,Bh,Ch,Dh,z,w] =  semhat(N); Ih=speye(N+1);

Rx = Ih(2:end-1,:);
Ry = Ih(2:end-1,:);

Mask = zeros(N+1,N+1); Mask(2:end-1,2:end-1)=1;




[z1,w1] = zwgll(N-1);
J1 = interp_mat(z1,z); % interpolate to GLL "midpoints"

[X,Y] = ndgrid(z,z);

Cx = -Y;   Cxm = J1*Cx*J1';
Cy =  X;   Cym = J1*Cy*J1';

dX = diff(X);  dX = dX*J1';         dUdx = Cxm./dX;
dY = diff(Y'); dY = dY'; dY=J1*dY;  dUdy = Cym./dY;

dUdx = max(max(abs(dUdx)+abs(dUdy)))

CFL    = 1.5
dtmax  = CFL/dUdx
nstep  = 2*pi/dtmax
nstep  = 32000
dt     = 2*pi/nstep

nsteps = 2*nstep 


X0 = 0.5; Y0 = 0.0;  delta = 0.1;
x = X-X0; y = Y-Y0; arg = -(x.*x+y.*y)/(delta^2);
U0 = exp(arg);

U = U0; dt2 = dt/2; dt6=dt/6;
emax = 0;
for istep=1:nsteps; time = dt*istep;

    k1 = -Mask.*( Cx.*(Dh*U ) + Cy.*(U *Dh') ); U1 = U + dt2*k1;
    k2 = -Mask.*( Cx.*(Dh*U1) + Cy.*(U1*Dh') ); U2 = U + dt2*k2;
    k3 = -Mask.*( Cx.*(Dh*U2) + Cy.*(U2*Dh') ); U3 = U + dt *k3;
    k4 = -Mask.*( Cx.*(Dh*U3) + Cy.*(U3*Dh') ); 
    U  = U + dt6*(k1 + 2*(k2 + k3) + k4);


    if mod(istep,20)==0;
       c=cos(time); s=sin(time);
       X0t =  c*X0 - s*Y0;
       Y0t =  s*X0 + c*Y0;
       x = X-X0t; y = Y-Y0t; arg = -(x.*x+y.*y)/(delta^2);
       Ue = exp(arg);
       Er = Ue-U;
       lmax = max(max(abs(Er)));
       emax = max(emax,lmax);
    end;

    if mod(istep,400)==0;
       mesh(X,Y,Er);
%      hold off; mesh(X,Y,U); hold on;  mesh(X,Y,Ue); 
       xlabel('X',fs,20); ylabel('Y',fs,20);
       title(['Time = ' num2str([time dt emax])],fs,15);
       drawnow;
    end;

end;

ktrial=ktrial+1;

kN(ktrial)=N;
kdt(ktrial)=dt;
kem(ktrial)=emax;
ktm(ktrial)=time;


