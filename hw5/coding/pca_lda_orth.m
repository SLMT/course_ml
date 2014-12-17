% Gaussian mean and covariance
d = 2;             % number of dimensions
num = 300;

% generate 100 samples
mean = [0, 0];
sigma = [0.1 0.2; 0.2 0.1];
sigma = sigma * sigma';
X1 = mvnrnd(mean, sigma, num);

% plot samples
scatter(X1(:,1), X1(:,2), 'o');

hold on;

% generate 100 samples
mean = [0.25, -0.25];
sigma = [0.1 0.2; 0.2 0.1];
sigma = sigma * sigma';
X1 = mvnrnd(mean, sigma, num);

% plot samples
scatter(X1(:,1), X1(:,2), 'o');

hold off;