function CdU=advects(U,Cr,Cs,Jh,Dt);

    CdU= Jh'*(Cr.*(Dt*U*Jh')+Cs.*(Jh*U*Dt'))*Jh;


   





