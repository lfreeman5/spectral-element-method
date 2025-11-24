function [x,iter,res,lam_max]=...
      pcg_lambda(Fl,tol,max_iter,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,dA,ifnull);

% Solve H u = f via CG,  H = b0*B + nu*A

hdr;

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

Mmax = size(M,1)*size(M,2)*size(M,3)
Mact = sum(sum(sum(M)))
ifnull=ifnull


%  In the code below, use sum(sum(sum(z.*r))) for z'*r.
%  In this way, we can keep z and r in "grid shape" either for 2D or 3D
%  We assume that r and w are _always_ unassembled, which allows easy
%  dot products in the redundant data layout


rho1=1; rtz1=1; r=Fl; x=0*r; p=x;

d=zeros(max_iter,1); l=d; u=d; %% For eigenvalue estimates

hold off;
for iter=1:max_iter;

   z=dA.*(M.*qqt(Q,r));   %% diagonal preconditioner
   rtz0 = rtz1; rtz1=sum(sum(sum(z.*r))); beta=rtz1/rtz0;
   res=sqrt(rtz1);
   if res < tol; break; end;
   if b0==0; semilogy(iter,res,'ro',lw,2); drawnow; hold on; end;
   p=z+beta*p;
   w=axl(p,b0,nu,Bl,Grr,Grs,Gss,Dh);
   rho0 = rho1; rho1 = sum(sum(sum(p.*w))); alpha = rtz1/rho1;
   x=x+alpha*p;
   r=r-alpha*w;

   if iter==1;
      d(iter)=rho1/rtz1;
   else;
      d(iter)  =(beta^2*rho0 + rho1)/rtz1;
      l(iter-1)=-beta*rho0/sqrt(rtz0*rtz1);
   end;
end;


if iter<max_iter; iter=iter-1; end;


d=d(1:iter); l=l(1:iter); u(2:iter)=l(1:iter-1); u=u(1:iter);
T=spdiags([l d u],-1:1,iter,iter); T=full(T);
d=eig(T); 
lam_min=min(d);
lam_max=max(d);


condition = 1;
if lam_min > 0; condition = lam_max / lam_min, pause(.1), end;

% format shorte;
% disp([iter res lam_min lam_max condition])

pause(.1)

