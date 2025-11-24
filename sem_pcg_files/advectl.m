function CdU=advectl(Ul,Cr,Cs,JM,DM);

Ul = tensor3(JM,1,JM,Ul);    % Interpolate U to fine mesh for dealiasing
Ur = tensor3(1,1,DM,Ul);     % du/dr: 
Us = tensor3(DM,1,1,Ul);     % du/ds: 

CdU= Cr.*Ur + Cs.*Us;        % C.grad(U) in Omega-hat
CdU= tensor3(JM',1,JM',CdU); % project back to coarse mesh


