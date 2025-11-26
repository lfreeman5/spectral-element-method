function se_mesh(X,Y,Z,name);
hdr

E=size(X,2);
N1=size(X,1);

% hold off;
for e=1:E
    x=X(:,e,:); x=reshape(x,N1,N1);
    y=Y(:,e,:); y=reshape(y,N1,N1);
    z=Z(:,e,:); z=reshape(z,N1,N1);
    mesh(x,y,z); hold on;
end;
title(name,fs,16);
xlabel('X',fs,16);
ylabel('Y',fs,16);
view(55,33);
