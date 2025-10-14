
%  2D Advection on Omega = [-1,1]

hdr; hold off;

N=100;

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
nstep  = 8000;
dt     = 2*pi/nstep

nsteps = 4*nstep 


X0 = 0.5; Y0 = 0.0;  delta = 0.1;
x = X-X0; y = Y-Y0; arg = -(x.*x+y.*y)/(delta^2);
U0 = exp(arg);

U = U0; dt2 = dt/2; dt6=dt/6;
for istep=1:nsteps; time = dt*istep;

    k1 = -Mask.*( Cx.*(Dh*U ) + Cy.*(U *Dh') ); U1 = U + dt2*k1;
    k2 = -Mask.*( Cx.*(Dh*U1) + Cy.*(U1*Dh') ); U2 = U + dt2*k2;
    k3 = -Mask.*( Cx.*(Dh*U2) + Cy.*(U2*Dh') ); U3 = U + dt *k3;
    k4 = -Mask.*( Cx.*(Dh*U3) + Cy.*(U3*Dh') ); 
    U  = U + dt6*(k1 + 2*(k2 + k3) + k4);


    if mod(istep,100)==0;
       c=cos(time); s=sin(time);
       Xt  =  c*X + s*Y;
       Yt  = -s*X + c*Y;
       x = Xt-X0; y = Yt-Y0; arg = -(x.*x+y.*y)/(delta^2);
       Ue = exp(arg);
       Er = Ue-U;
       mesh(X,Y,Er);
%      hold off; mesh(X,Y,U); hold on;  mesh(X,Y,Ue); 
       drawnow;
    end;

end;

