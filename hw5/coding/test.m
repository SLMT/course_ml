
clear all; clc;

load data.dat

y = model.clustering.KmeansClustering.cluster(data, 2);

group1 = data(y(:, 1) == 1, :);
group2 = data(y(:, 2) == 1, :);
plot(group1(:, 1), group1(:, 2), '.g', group2(:, 1), group2(:, 2), '.r');
