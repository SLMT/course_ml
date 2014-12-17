% Gaussian mean and covariance
d = 2;             % number of dimensions
num = 300;

% generate 100 samples
mean = [0, 0];
sigma = [0.1 0.9; 0.9 0.1];
sigma = sigma * sigma';
X1 = mvnrnd(mean, sigma, num);

% plot samples
scatter(X1(:,1), X1(:,2), 'o');

hold on;

% generate 100 samples
mean = [3, -3];
sigma = [0.1 0.98; 0.98 0.1];
sigma = sigma * sigma';
X1 = mvnrnd(mean, sigma, num);

% plot samples
scatter(X1(:,1), X1(:,2), 'o');

hold off;