function [Ah,Bh,Ch,Dh,Ih,J,z,w,Jf,zf] = lfsemhat(N,ifourier);  % Set basic operators

M =ceil(1.3*N);   %% For dealiasing advection
Mf=ceil(5+2.3*N); %% For plotting advection

Ih=speye(N+1);

if ifourier > 0;   %% Fourier

      [Ah,Bh,Ch,Dh,z,w] = fsemhat(N);

      [zo,wo]=zwuni(M);   % for dealiasing
      [zf,wf]=zwuni(Mf);  % for plotting

      J  = f_interp_mat(zo,z);
      Jf = f_interp_mat(zf,z);

else;             %% Legendre

      [Ah,Bh,Ch,Dh,z,w] = semhat(N);

      [zo,wo]=zwgl(M);    % for dealiasing
      [zf,wf]=zwuni(Mf);  % for plotting

      J  = interp_mat(zo,z);
      Jf = interp_mat(zf,z);

end;

