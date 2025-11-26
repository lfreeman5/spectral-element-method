function se_disp(X,name);
hdr

E=size(X,2);
N1=size(X,1);

for e=1:E
    disp([name ' e=' int2str(e)])
    t=X(:,e,end:-1:1); t=reshape(t,N1,N1);
    disp(t')
end;
