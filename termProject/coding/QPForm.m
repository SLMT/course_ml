
n = size(X, 1);
d = size(X, 2);
c = 0.1;

% set QP form
QPn = d + n + 1;

QPQ = zeros(QPn);
QPQ(1:d, 1:d) = eye(d);

QPc = zeros(QPn, 1);
QPc(d+1:d+n) = c;

QPG = zeros(n + n, QPn);
tmp = X;
for i = 1 : n
    tmp(i,:) = tmp(i, :) * -y(i);
end
QPG(1:n, 1:d) = tmp;
QPG(1:n, d+1:d+n) = eye(n);
QPG(1:n, d+n+1) = y;
QPG(n+1:n+n, d+1:d+n) = eye(n);

QPh = zeros(n + n, 1);
QPh(1:n) = -1;

cvx_begin
    variable QPx(QPn)
    minimize( QPx' * QPQ * QPx + QPc' * QPx )
    subject to
        QPG * QPx <= QPh
cvx_end