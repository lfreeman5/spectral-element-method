function [x,res,iter]=mg_ksp(Fl,lam_max,omega,tol,iter_max,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,dA,ifnull,...
                                                   Jc,Mc,Qc,Blc,Grrc,Grsc,Gssc,Dc,dAc);

rl=Fl; Umg=0*Fl;

vol = sum(sum(sum(Blc)));
for iter=1:iter_max;

  %% Smoothing step
  if omega > 0;
    [Us]=smoother(rl,lam_max,omega,5,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,dA,ifnull);
  else;
    [Us]=cheby4(rl,lam_max,omega,6,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,dA,ifnull);
  end;
  Umg = Umg+Us;

  %% Coarse-grid correction
  rl = rl-axl(Us,b0,nu,Bl,Grr,Grs,Gss,Dh);
  rc = tensor3(Jc',1,Jc',rl);
  zc = (Blc.*Mc).*(dAc.*rc).^2; zcn=sqrt(sum(sum(sum(zc)))/vol); 
  
  tolc=0.1*zcn;
  [Ec,iterc,res,lam_crs]=...
      pcg_lambda(rc,tolc,100,b0,nu,Mc,Qc,Blc,Grrc,Grsc,Gssc,Dc,dAc,ifnull);
  El = tensor3(Jc,1,Jc,Ec);

  Umg = Umg + El;
  rl  = rl-axl(El,b0,nu,Bl,Grr,Grs,Gss,Dh);

  zl = (Bl.*M).*(dA.*rl).^2; zn=sqrt(sum(sum(sum(zl)))/vol); 

  if zn < tol; break; end;

end;

res = zn;
iter=min(iter,iter_max);


