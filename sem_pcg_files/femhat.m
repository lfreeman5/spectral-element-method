      function[Ah,Bh,Ch,Dh,z,w] =  femhat(N,igll)
%
%     1D fem Stiffness, Mass, and Convection matrix analogies to semhat
%

      if igll==1;
         [z,w] = zwgll(N); 
      else;
         [z,w] = zwuni(N);
      end;

      consistent=1;  %% Will produce zeros on diagonal of Dh.

      n  = N+1;

      Ah=zeros(n,n); Bh=Ah; Ch=Ah; Dh=Ah; Bf=Ah;

      for e=1:N; i0=e; i1=e+1;
          h=z(i1)-z(i0); hi=1./h;
          Ah(i0,i0) = Ah(i0,i0) + hi;
          Ah(i0,i1) =           - hi;
          Ah(i1,i0) =           - hi;
          Ah(i1,i1) =             hi;

          Bf(i0,i0) = Bf(i0,i0) + 2*h/3;
          Bf(i0,i1) =               h/3;
          Bf(i1,i0) =               h/3;
          Bf(i1,i1) =             2*h/3;

          Bh(i0,i0) = Bh(i0,i0) + h;
          Bh(i1,i1) =             h;

          Ch(i0,i0) = Ch(i0,i0) - 1;
          Ch(i0,i1) =             1;
          Ch(i1,i0) =           - 1;
          Ch(i1,i1) =             1;

          if consistent==1;
             Dh(i0,i0) = Dh(i0,i0) - 1;
             Dh(i0,i1) =             1;
             Dh(i1,i0) =           - 1;
             Dh(i1,i1) =             1;
          else;
             Dh(i0,i0) = Dh(i0,i0) - hi;
             Dh(i0,i1) =             hi;
             Dh(i1,i0) =           - hi;
             Dh(i1,i1) =             hi;
          end;
      end;


Ah=sparse(Ah+Ah')/2;
Bh=sparse(Bh);
Ch=sparse(Ch);
Dh=sparse(Dh);
w = diag(Bh);

if consistent==1; Dh=Bh\Dh; end;

Ah=full(Ah+Ah')/2;
Bh=full(Bh);
Ch=full(Ch);
Dh=full(Dh);

