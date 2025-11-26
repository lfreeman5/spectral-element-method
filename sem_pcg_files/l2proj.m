function U0=l2proj(Ul,Q,M,Bl);

B  = qqt(Q,Bl);
U0 = qqt(Q,M.*Bl.*Ul)./B;

