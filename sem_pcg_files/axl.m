function Wl = axl(Ul,b0,nu,Bl,Grr,Grs,Gss,Dh);

%
%   A = D^T G D
%

% Ul  - rhs, in grid form (full grid)
% Gij - geometric factors, in grid form
% Dhr - derivative in Omega-hat, r direction
% Dhs - derivative in Omega-hat, s direction

Ur = tensor3(1,1,Dh,Ul);              % dU/dr: 
Us = tensor3(Dh,1,1,Ul);              % dU/ds: 

Wr = nu.*(Grr.*Ur + Grs.*Us);  % Can support variable nu
Ws = nu.*(Grs.*Ur + Gss.*Us);
Wl = tensor3(1,1,Dh',Wr) + tensor3(Dh',1,1,Ws) + b0.*(Bl.*Ul);


