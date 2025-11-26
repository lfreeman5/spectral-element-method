function [W]=qqt_op(U,op,glo_num);

E =size(U,2);
N1=size(U,1);

nL=N1*N1*E;

U=reshape(U,nL,1);

if op=='+'; 
   W=0*U;
   for il=1:nL; ig=glo_num(il);
       W(ig)=W(ig)+U(il);
   end;
end;

if op=='*'; 
   W=0*U + 1;
   for il=1:nL; ig=glo_num(il);
       W(ig)=W(ig)*U(il);
   end;
end;

if op=='m'; 
   W=0*U + 1e30;
   for il=1:nL; ig=glo_num(il);
       W(ig)=min(W(ig),U(il));
   end;
end;

if op=='M'; 
   W=0*U - 1e30;
   for il=1:nL; ig=glo_num(il);
       W(ig)=max(W(ig),U(il));
   end;
end;

for il=1:nL; ig=glo_num(il);
    T(il)=W(ig);
end;

W=reshape(T,N1,E,N1);
