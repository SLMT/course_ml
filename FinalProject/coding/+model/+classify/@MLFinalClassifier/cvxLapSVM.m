function [ y_predicted ] = cvxLapSVM( y, K, S, miu, lambda, silent )
% Using CVX to solve LapSVM
n = size(y,1);
l = nnz(y); %Number of Labeled data
labeled = y~=0;

D = diag(sum(S, 2));
L = D - S;

% Use CVX to solve Objective
c = zeros(n, 1);

if silent
cvx_begin quiet
else
cvx_begin
end
	variables c(n)
	variable xi(l) nonnegative
	
				  %Data term    %Smoothness term   %Regularization Term
	minimize( norm(xi,1) + miu*(K*c)'*L*(K*c) + lambda*c'*K*c)
	
	subject to
		%Count the slack of labeled data only
		y(labeled) .* (  K(labeled,labeled) * c(labeled)  ) >= 1-xi
	
cvx_end

y_predicted = K * c;

end

