function [W]=qqt(Q,U);

E =size(U,2);
N1=size(U,1);

nL=N1*N1*E;

W=Q*(Q'*reshape(U,nL,1));
W=reshape(W,N1,E,N1);

