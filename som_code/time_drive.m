
jtrial = 0;

for N=20:10:110;
   space_chk
end;

figure; 
semilogy(jN,jem,'r.',ms,20,jN,jem,'k--')
axis square
title('Spatial Convergence: $t_{\mbox{final}}=2\pi, \; \Delta t=\pi/8000$',intp,ltx,fs,14);
xlabel('$N$',intp,ltx,fs,20); 
ylabel('$\| {\tilde u} - u \|_{\infty}^{}$',intp,ltx,fs,20)
