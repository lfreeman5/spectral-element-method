function [curlcurlX,curlcurlY,Omega]=curlcurl(U,V,Bl,Rx,Dh);

%%   Evaluate curl-curl term (to be extrapolated)

     Omega     = tensor3(1,1,Dh,V)-tensor3(Dh,1,1,U);
%    curlcurlX =  Bl.*(Lyi*(Omega*Dhy'));
%    curlcurlY = -Bl.*(Lxi*(Dhx*Omega));
     curlcurlX =  Omega;
     curlcurlY = -Omega;

% Ur = tensor3(1,1,Dh,U);     % du/dr: 
% Us = tensor3(Dh,1,1,U);     % du/ds: 
% Vr = tensor3(1,1,Dh,V);     % du/dr: 
% Vs = tensor3(Dh,1,1,V);     % du/ds: 
% 
% rx = Rx(:,:,:,1,1);  % dr/dx
% ry = Rx(:,:,:,1,2);  % dr/dy
% sx = Rx(:,:,:,2,1);  % ds/dx
% sy = Rx(:,:,:,2,2);  % ds/dy
% 
% Ux = Ur.*rx + Us.*sx;
% 
