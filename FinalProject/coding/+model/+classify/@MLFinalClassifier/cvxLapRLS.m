function [ y_predicted ] = cvxLapRLS( y, K, S, miu, lambda, silent )

n = size(y,1);

D = diag(sum(S, 2));
L = D - S;

% Compute A (To eliminate unlabeled data)
A = diag (double( y~=0 )) ;

% Use CVX to solve Objective
c = zeros(n, 1);

if silent
cvx_begin quiet
else
cvx_begin
end
	variable c(n)
	minimize((y - K * c)' * A * (y - K * c) + miu * (K * c)' * L * (K * c) + lambda * c' * K * c)
cvx_end

y_predicted = K * c;

end

