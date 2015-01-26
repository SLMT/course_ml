function [ K ] = getGaussianKernel( X, gamma )
% Compute an N¡ÑN Gaussian Kernal for X

n = size(X,1);

K = zeros(n, n);
for i = 1 : n
	for j = i : n
		result = exp(- gamma * norm(X(i, :) - X(j, :)) ^ 2);
		
		% fill in result
		K(i, j) = result;
		K(j, i) = result;
	end
end


end

