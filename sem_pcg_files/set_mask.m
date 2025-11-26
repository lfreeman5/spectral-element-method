function [mask,Q,glo_num]=set_mask(BC,Nelx,Nely,Q,glo_num)

hdr;

bc_left =BC(1); bc_right=BC(2);
bc_lower=BC(3); bc_upper=BC(4);


N1=size(glo_num,1); N=N1-1;
E =Nelx*Nely;

mask = ones(N1,Nelx,Nely,N1);

if bc_left =='D'; mask( 1,   1,:,:)=0; end;
if bc_right=='D'; mask(N1,Nelx,:,:)=0; end;
if bc_lower=='D'; mask(:,:,   1, 1)=0; end;
if bc_upper=='D'; mask(:,:,Nely,N1)=0; end;
mask=reshape(mask,N1,E,N1);

if bc_left=='P' || bc_lower=='P';

   glo_num = reshape(glo_num,N1,Nelx,Nely,N1);

   if bc_left=='P';  glo_num(end,end,:,:)=glo_num(1,1,:,:); end;
   if bc_lower=='P'; glo_num(:,:,end,end)=glo_num(:,:,1,1); end;

   glo_num = reshape(glo_num,N1*Nelx*Nely*N1,1);
   [g,ia,glo_num] = unique(glo_num);
   glo_num = reshape(glo_num,N1,Nelx*Nely,N1);

   nL=N1*Nelx*Nely*N1;
   Q=speye(nL); 
   k=0;
   for j=1:N1;
   for e=1:E;
   for i=1:N1;
       ig = glo_num(i,e,j);
       k  = k+1;
       Q(k,k)=0; Q(k,ig)=1;
   end;
   end;
   end;
   n=max(max(max(glo_num)));
   Q=Q(:,1:n);

end;
