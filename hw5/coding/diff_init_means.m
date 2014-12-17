% initialize variables
n = 100;
k = 3;
X = zeros(n * k, 2);

% define ranges
range = zeros(k, 4);
range(1, :) = [1 2 2 4.5];
range(2, :) = [1 2 4.5 7];
range(3, :) = [4 5 4 5];

for i = 1 : k
    left = range(i, 1);
    right = range(i, 2);
    down = range(i, 3);
    up = range(i, 4);
    
    startI = (i-1) * n + 1;
    endI = i * n;
    X(startI:endI, 1) = left + (right - left) * rand(1, n);
    X(startI:endI, 2) = down + (up - down) * rand(1, n);
end

% === Bad Init Means ===

% init means
means(1, :) = X(randsample(1:n, 1), :);
means(2, :) = X(randsample(n + 1 : 2 * n, 1), :);

% k-means
[y, finalMeans] = model.clustering.KmeansClustering.performClustering(X, 2, means);

% plot
subplot(1, 2, 1);

group1 = X(y(:, 1) == 1, :);
group2 = X(y(:, 2) == 1, :);
plot(group1(:, 1), group1(:, 2), '.g', group2(:, 1), group2(:, 2), '.r', means(:, 1), means(:, 2), 'bo', finalMeans(:, 1), finalMeans(:, 2), 'mo');
axis([0, 6, 1, 8]);

% === Good Init Means ===

% init means
means(1, :) = X(randsample(n + 1 : 2 * n, 1), :);
means(2, :) = X(randsample(2 * n + 1 : 3 * n, 1), :);

% k-means
[y, finalMeans] = model.clustering.KmeansClustering.performClustering(X, 2, means);

% plot
subplot(1, 2, 2);

group1 = X(y(:, 1) == 1, :);
group2 = X(y(:, 2) == 1, :);
plot(group1(:, 1), group1(:, 2), '.g', group2(:, 1), group2(:, 2), '.r', means(:, 1), means(:, 2), 'bo', finalMeans(:, 1), finalMeans(:, 2), 'mo');
axis([0, 6, 1, 8]);