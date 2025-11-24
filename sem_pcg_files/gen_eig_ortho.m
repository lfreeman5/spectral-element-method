function [S,Lam]=gen_eig_ortho(A,B);

%
%  Return matrix of eigenvectors,   S   = [s1 ... sn],
%     and matrix of eigenvalues,    Lam = diag(lam_k).
%
%  The columns of S are the eigenvectors, sk, k=1,...,n
%  Lam is defined as a sparse matrix.
%
%  The eigenvectors are orthonormalized w.r.t. the B inner-product:
%
%
%  sj^T B sk = delta_jk
%

%   Below is original code
%
%%  [S,Lam]=eig(A,B); d=diag(D);
%%  [d,ind]=sort(d); S=S(:,ind);
%%  
%%  n=length(d);
%%  for k=1:n
%%      for j=1:k-1;
%%          beta = S(:,k)'*B*S(:,j);
%%          S(:,k) = S(:,k) - beta*S(:,j);
%%      end;
%%      beta = S(:,k)'*B*S(:,k);
%%      S(:,k) = S(:,k)/sqrt(beta);
%%  end;
%%  
%%  Lam=diag(d); Lam=sparse(Lam);

%   Below is new code provided by James Lottes
%
%   Generalized symmetric eigendecomposition, normalized

A=full(A+A')/2;
B=full(B+B')/2;
% [S,Lam] = eig(A,B,'chol');  %% This doesn't seem to work for Octave
[S,Lam] = eig(A,B);
lambda = diag(Lam);
[lambda, i] = sort(lambda);
S = S(:, i);
S = S / chol(S' * B * S);
S = S / chol(S' * B * S);  % This orthogonalization may not be necessary

Lam = lambda;  %% Return as a vector

% Lam=diag(lambda); Lam=sparse(Lam);

