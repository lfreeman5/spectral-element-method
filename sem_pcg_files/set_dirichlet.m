function [U,V,T] = set_dirichlet(Uo,Vo,To,Mu,Mv,Mt,X,Y);

U = 1 - Y.*Y;  %% Desired field at inflow
V = 0 + 0*X;  %% Desired field at inflow
T = 1 + 0*X;  %% Desired field at inflow

U = Mu.*Uo + (1-Mu).*U;   %% Old value in interior, new value on inflow
V = Mv.*Vo + (1-Mv).*V;
T = Mt.*To + (1-Mt).*T;

% se_mesh(X,Y,U,'U bdry'); pause(1); pause



























