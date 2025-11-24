function [Px,Py] = grad(P,Rx,Dh);

%%   Evaluate gradient of P using chain rule

%    divUt     = tensor3(1,1,Dh,V)-tensor3(Dh,1,1,U);

Pr = tensor3(1,1,Dh,P);
Ps = tensor3(Dh,1,1,P);

Px = Rx(:,:,:,1,1).*Pr + Rx(:,:,:,2,1).*Ps;
Py = Rx(:,:,:,1,2).*Pr + Rx(:,:,:,2,2).*Ps;

