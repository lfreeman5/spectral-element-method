% 2-D SEM convection on periodic domain
close all; format compact

lw='linewidth';               %%% Plotting defs
fs='fontsize';                %%% Plotting defs
intp = 'interpreter';         %%% Plotting defs
ltx  = 'latex';               %%% Plotting defs




[Ah,Bh,Ch,Dh,z,w] = semhat(N);

nh = N+1; Ih = speye(nh); R = Ih(2:N-1,:);

x = z;  
y = z;  

[X,Y] = ndgrid(x,y);


 tfinal = 2*pi;

Cx = -Y; Cy = X; 

delta = 0.10;
Yt = Y+.5;
R2 = X.*X + Yt.*Yt;
U0 = exp(-(R2/(delta*delta)).^1);
U  = U0;

Ux = Dh*U; mesh(X,Y,Ux); 

 dx   = min(diff(x));
 c    = 1;
 CFL  = .15;
 if c>0;  dt = CFL*(dx/c);   end; 
 if c==0; dt = CFL*dx;       end; 
 nsteps = floor(tfinal/dt);
 iostep = floor(nsteps/40);
 nsteps = iostep*ceil(nsteps/iostep);
 dt     = tfinal/nsteps;

 Ux = Dh*U; Uy = U*Dh';
 f0 = R* ( Bh*( Cx.*Ux + Cy.*Uy )*Bh') *R' ;
 f1 = 0*f0; 
 f2 = 0*f0; 

 B = R*Bh*R'; Bi = sparse(inv(B));

 k=0;
 tstart=tic;
 for iloop=1:1;
    for istep =1:nsteps; k=k+1;

       if k==1; 
          c0 = -dt;
          c1 =   0;
          c2 =   0;
       elseif k==2; 
          c0 = -dt*1.5;
          c1 =  dt*0.5;
          c2 =   0;
       else;
          c0 = -dt*23./12.;
          c1 =  dt*16./12.;
          c2 = -dt*5.0/12.;
       end;

       f2 = f1; f1 = f0; 
       Ux = Dh*U; Uy = U*Dh';
       f0 = R* ( Bh*( Cx.*Ux + Cy.*Uy )*Bh') *R' ;
       U  = U + R'*Bi*(c0*f0 + c1*f1 + c2*f2)*Bi'*R;

       time = k*dt;
       if mod(istep,iostep)==0; mesh(X,Y,U); drawnow; pause(.1); end;
    end;
    figure; mesh(X,Y,U); title('demo-2d.m',fs,18); drawnow; 
    figure; mesh(X,Y,U0-U); title('demo-2d.m',fs,18); drawnow; 
 end;
 elapsed_time = toc(tstart);

