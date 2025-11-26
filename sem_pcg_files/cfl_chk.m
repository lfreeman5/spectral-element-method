function CdU=advects(U,Cr,Cs,Jh,Dt);

[
dx=diff




    CdU= Jh'*(Cr.*(Dt*U*Jh')+Cs.*(Jh*U*Dt'))*Jh;


function Wl = axl(Ul,b0,nu,Bl,Grr,Grs,Gss,Dh);

%
%   A = D^T G D
%

% Ul  - rhs, in grid form (full grid)
% Gij - geometric factors, in grid form
% Dhr - derivative in Omega-hat, r direction
% Dhs - derivative in Omega-hat, s direction

Ur = tensor3(1,1,Dh,Ul);              % dx/dr: 
Us = tensor3(Dh,1,1,Ul);              % dx/ds: 

Wr = nu.*(Grr.*Ur + Grs.*Us);  % Can support variable nu
Ws = nu.*(Grs.*Ur + Gss.*Us);
Wl = tensor3(1,1,Dh',Wr) + tensor3(Dh',1,1,Ws) + b0.*(Bl.*Ul);


   





uuuu

