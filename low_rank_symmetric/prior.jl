#= Gaussian with var_gauss variance =#
function f_gauss(A,B,var_gauss=1)
   VAR=inv((1 ./var_gauss)*eye(A)+A);
   MEAN=B*VAR;
   logZ=-0.5*log(det(var_gauss*eye(A)+A))*size(B,1)+trace(0.5*B'*B*VAR);   
   MEAN,VAR,logZ;
end