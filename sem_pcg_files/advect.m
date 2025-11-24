function CdU=advect(U,Cr,Cs,Jr,Js,Dr,Ds);

    CdU= Jr'*(Cr.*(Dr*U*Js')+Cs.*(Jr*U*Ds'))*Js;


   





