#= Gaussian with var_gauss variance =#
# B: vector rx1
# A: matrix rxr
using LinearAlgebra

function matrix_trace(A)
   return sum(diag(A))
end

function random_positive_definite_matrix(n::Int)
   A = randn(n, n)
   return A' * A + n * I
end

function f_gauss(A,B,var_gauss=1)
   VAR=inv((1 ./var_gauss)*Matrix(I, size(A)[1], size(A)[1])+A);
   MEAN=B'*VAR;
   logZ=-0.5*log(det(var_gauss*Matrix(I, size(A)[1], size(A)[1])+A))*size(B,1)+matrix_trace(0.5*B*B'*VAR);   
   MEAN,VAR,logZ;
end

# other functions f_clust, f_Rank1Binary
r = 5
B = reshape(1:r, r, 1)

# columnwise entry
#A = reshape(1:25, r, r)

# rowise entry
# A = permutedims(reshape(1:25, r, r))


A = random_positive_definite_matrix(r)


f_gauss(A,B,var_gauss=1)