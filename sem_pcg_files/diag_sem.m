function dAL = diag_sem(Grr,Grs,Gss,Dh);

dAL = 0*Grr;

E=size(Grr,2);
for e=1:E;
    grr=Grr(:,e,:);
    grs=Grs(:,e,:);
    gss=Gss(:,e,:);
    da = diag_a(grr,grs,gss,Dh,Dh);
    dAL(:,e,:)=da;
end;

% dA=qqt(Q,dAL) - this works if you have no "ghost" elements

