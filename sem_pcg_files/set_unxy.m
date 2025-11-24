function [unxa_v,unya_v] = set_unxy(mask,X,Y,Xr);

%
%  This is a very _un_general way to set the unit normals for the pressure 
%  Poisson solve.
%
%  It won't work, for example, and element has 
%  inflow boundary conditions on adjacent faces 
%
%  Normally, we would (correctly) put the unit normals into a surface array
%
%  But this approach is expeditious and OK for the HW assignment

hdr


unxa_v = 0*X;
unya_v = 0*X;

dxdr = Xr(:,:,:,1,1);
dxds = Xr(:,:,:,1,2);
dydr = Xr(:,:,:,2,1);
dyds = Xr(:,:,:,2,2);

N1=size(X,1); N=N1-1;
E =size(X,2);

[z,w]=zwgll(N);

for e=1:E;

%% Check if "left" face is Dirichlet:

  if mask(1,e,1)==0 && mask(1,e,end)==0; 
    tan_x=reshape(dxds(1,e,:),N1,1);
    tan_y=reshape(dyds(1,e,:),N1,1);
    area = sqrt(tan_x.^2+tan_y.^2);
    unx  = -tan_y./area;
    uny  =  tan_x./area;
    unxa_v(1,e,:) = unx.*area.*w;  %% unxa = unx * area !
    unya_v(1,e,:) = uny.*area.*w;
  end;

%% Check if "right" face is Dirichlet:

  if mask(end,e,1)==0 && mask(end,e,end)==0; 
    tan_x=reshape(dxds(end,e,:),N1,1);
    tan_y=reshape(dyds(end,e,:),N1,1);
    area = sqrt(tan_x.^2+tan_y.^2);
    unx  = -tan_y./area;
    uny  =  tan_x./area;
    unxa_v(end,e,:) = unx.*area.*w;
    unya_v(end,e,:) = uny.*area.*w;
  end;
   
%% Check if "lower" face is Dirichlet:

  if mask(1,e,1)==0 && mask(end,e,1)==0; 
    tan_x=reshape(dxdr(:,e,1),N1,1);
    tan_y=reshape(dydr(:,e,1),N1,1);
    area = sqrt(tan_x.^2+tan_y.^2);
    unx  = -tan_y./area;
    uny  =  tan_x./area;
    unxa_v(:,e,1) = unx.*area.*w;
    unya_v(:,e,1) = uny.*area.*w;
  end;

%% Check if "upper" face is Dirichlet:

  if mask(1,e,end)==0 && mask(end,e,end)==0; 
    tan_x=reshape(dxdr(:,e,end),N1,1);
    tan_y=reshape(dydr(:,e,end),N1,1);
    area = sqrt(tan_x.^2+tan_y.^2);
    unx  = -tan_y./area;
    uny  =  tan_x./area;
    unxa_v(:,e,end) = unx.*area.*w;
    unya_v(:,e,end) = uny.*area.*w;
  end;

% str=['unx x area: ' int2str(e)];
% se_mesh(X,Y,unxa_v,str); pause(1); pause

% str=['uny x area: ' int2str(e)];
% se_mesh(X,Y,unya_v,str); pause(1); pause

end;


