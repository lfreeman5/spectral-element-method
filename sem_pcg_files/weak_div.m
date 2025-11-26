function divUt = weak_div(U,V,Bl,Rx,Dh);

%%   Evaluate weak (i.e., tranposed) divergence-operator

%    divUt     = tensor3(1,1,Dh,V)-tensor3(Dh,1,1,U);

wx=Bl.*U; wy=Bl.*V;

rx = Rx(:,:,:,1,1);  % dr/dx
ry = Rx(:,:,:,1,2);  % dr/dy
sx = Rx(:,:,:,2,1);  % ds/dx
sy = Rx(:,:,:,2,2);  % ds/dy

wr = rx.*wx + ry.*wy;
ws = sx.*wx + sy.*wy;

divUt = tensor3(1,1,Dh',wr) +tensor3(Dh',1,1,ws);


