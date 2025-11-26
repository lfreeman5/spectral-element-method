      function[p] =  phip(x,N)
%
%     Compute the bubble functions up to degree N
%
%     This might not be the best ordering of the functions if you wish to 
%     use them as a basis.
%
      [p]=legendp(x,N);
%
      for k=N:-1:3;
          i=k+1;
          p(:,i) = p(:,i)-p(:,i-2);
      end;

