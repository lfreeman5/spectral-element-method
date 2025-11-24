function [x]=cheby4(Fl,lam_max,omega,k,b0,nu,M,Q,Bl,Grr,Grs,Gss,Dh,Di,ifnull);

%% Apply kth-order optimized 4th-kind Chebyshev (from Malachi Phillips dissertation,2023).

%% From Table 5 of Malachi Phillips paper

%% NEED A WARNING IF k > 7, because table is incomplete.

Beta=[ ...
1.12500000000000                0                0                0                0                0                0;
1.02387287570313 1.26408905371085                0                0                0                0                0;
1.00842544782028 1.08867839208730 1.33753125909618                0                0                0                0;
1.00391310427285 1.04035811188593 1.14863498546254 1.38268869241000                0                0                0;
1.00212930146164 1.02173711549260 1.07872433192603 1.19810065292663 1.41322542791682                0                0;
1.00128517255940 1.01304293035233 1.04678215124113 1.11616489419675 1.23829020218444 1.43524297106744                0;
1.00083464397912 1.00843949430122 1.03008707768713 1.07408384092003 1.15036186707366 1.27116474046139 1.45186658649364];


lmax_i = 1./lam_max;

beta=Beta(k,:);
r = Fl;
x = 0*r;
d = (lmax_i*4./3.)*(Di.*(M.*qqt(Q,r)));%Diagonal preconditioner

for i=1:k-1;
   x = x + beta(i)*d;
   r = r - axl(d,b0,nu,Bl,Grr,Grs,Gss,Dh);
   s1= (2*i-1)/(2*i+3); s2=lmax_i*(8*i+4)/(2*i+3);
   d = s1*d + s2*Di.*(M.*qqt(Q,r));
end;
x = x + beta(k)*d;

