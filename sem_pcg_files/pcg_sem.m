function [x,iter,residual]=pcg_sem(Fl,tol,max_iter,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,ifnull);

% Solve H u = f via CG,  H = b0*B + nu*A

%
%   A = D^T G D
%

% Fl  - rhs, unassembled, in grid form (full grid)
% Gij - geometric factors, in grid form
% Dh  - derivative in Omega-hat
% M   - Mask, used to restrict to H^1_0
%
% ifnull > 0 --- Implies that H has a null space of 1 (not yet coded)
%


%  In the code below, use sum(sum(sum(z.*r))) for z'*r.
%  In this way, we can keep z and r in "grid shape" either for 2D or 3D
%  We assume that r and w are _always_ unassembled, which allows easy
%  dot products in the redundant data layout


rho1=1; r=Fl; x=0*r; p=x;

for iter=1:max_iter;

   z=M.*qqt(Q,r);   %% Need a preconditioner here; project into continuous space
   rho0 = rho1; rho1=sum(sum(sum(z.*r))); beta=rho1/rho0;
   residual=sqrt(rho1);
   if residual < tol; break; end;
   p=z+beta*p;
   w=axl(p,b0,nu,Bl,Grr,Grs,Gss,Dh);
   gamma = sum(sum(sum(p.*w))); alpha = rho1/gamma;
   x=x+alpha*p;
   r=r-alpha*w;

end;

if iter<max_iter; iter=iter-1; end;

  pcg_sem_results = disp([iter residual])


