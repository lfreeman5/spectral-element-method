
function [x]=smoother...
   (Fl,lam_max,omega,m,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,Di,ifnull);

% Apply m rounds of Jacobi smoothing


omega = omega * (2 / lam_max);

r = Fl;
z = Di.*(M.*qqt(Q,r));   %% diagonal preconditioner
x = omega*z;             %% initial guess for x is 0.

for iter=2:m;
   r = Fl - axl(x,b0,nu,Bl,Grr,Grs,Gss,Dh);
   z = Di.*(M.*qqt(Q,r));
   x = x + omega*z;
end;


