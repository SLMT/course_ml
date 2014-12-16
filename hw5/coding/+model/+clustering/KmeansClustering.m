
% Written by SLMT
classdef KmeansClustering
    methods (Static)
        function y = cluster (X, k)
            % initialize variables
            m = size(X, 1);
            n = size(X, 2);
            means = zeros(k, n);
            groups = zeros(m, 1);
            
            % initialize means (k-means++)
            weights = ones(m, 1);
            for newMeanIndex = 1 : k
                % update mean point
                nextMean = randsample(m, 1, true, weights);
                weights(nextMean) = 0;
                means(newMeanIndex, :) = X(nextMean, :);
                
                % update weights (can be optimized)
                for dataIndex = 1 : m
                    
                    if weights(dataIndex) ~= 0 % it means that this point is not a mean point
                        data = X(dataIndex, :);
                        nearestDis = norm(data - means(1, :));

                        for meanIndex = 2 : newMeanIndex
                            dis = norm(data - means(meanIndex, :));
                            if dis < nearestDis
                                nearestDis = dis;
                            end
                        end

                        weights(dataIndex) = nearestDis ^ 2;
                    end
                end
            end
            
            % k-means algorithm
            change = 1;
            while change ~= 0
                % initialize conditions
                change = 0;
                
                % find nearest group for each data point
                for dataIndex = 1 : m
                    data = X(dataIndex, :);
                    nearestGroup = 1;
                    nearestDis = norm(data - means(1, :));

                    for meanIndex = 2 : k
                        dis = norm(data - means(meanIndex, :));
                        if dis < nearestDis
                            nearestGroup = meanIndex;
                            nearestDis = dis;
                        end
                    end
                    
                    % check change
                    if change == 0 && groups(dataIndex) ~= nearestGroup
                        change = 1;
                    end
                    
                    % update group
                    groups(dataIndex) = nearestGroup;
                end
                
                % update means accroding to new groups
                for meanIndex = 1 : k
                    groupData = X(groups == meanIndex);
                    means(meanIndex) = sum(groupData) / size(groupData, 1);
                end
            end

            % generate labels
            y = zeros(m, k);
            for dataIndex = 1 : m
                y(dataIndex, groups(dataIndex)) = 1;
            end
        end
    end
end

