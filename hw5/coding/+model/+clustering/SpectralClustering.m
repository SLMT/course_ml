classdef SpectralClustering

	methods (Static)
        function Y = cluster (X, k, cfg)
            % initialize variables
            m = size(X, 1);
            extremeLowNum = 0.0000000001;
            
            % generate similarity matrix
            S = zeros(m, m);
            if strcmp(cfg('similarity'), 'eNN')
                % initialize hyperparameters
                epslion = 80;
                
                % compute distance matrix
                distance = zeros(m, m);
                for i = 1 : m
                    for j = i : m
                        result = norm(X(i, :) - X(j, :));

                        % fill in result
                        distance(i, j) = result;
                        distance(j, i) = result;
                    end
                end
                
                % get nearest elements
                [~, Index] = sort(distance);
                
                % compute similarity matrix
                for i = 1 : m
                    for j = 1 : m
                        % fill in result
                        if Index(i, j) <= epslion
                            S(i, j) = distance(i, j);
                        else
                            S(i, j) = extremeLowNum;
                        end
                    end
                end
                
            elseif strcmp(cfg('similarity'), 'eBall')
                % initialize hyperparameters
                epslion = 1;
                
                % compute matrix
                for i = 1 : m
                    for j = i : m
                        result = norm(X(i, :) - X(j, :));
                        if result > epslion
                            result = extremeLowNum;
                        end
                        
                        % fill in result
                        S(i, j) = result;
                        S(j, i) = result;
                    end
                end
                
            elseif strcmp(cfg('similarity'), 'Gaussain')
                % initialize hyperparameters
                sigma = 0.5;
                
                % compute each element
                for i = 1 : m
                    for j = i : m
                        result = exp(- norm(X(i, :) - X(j, :)) ^ 2 / sigma ^ 2);
                        
                        % fill in result
                        S(i, j) = result;
                        S(j, i) = result;
                    end
                end
            end
            
            % compute L
            D = diag(sum(S, 2));
            L = D - S;
            
            % compute eigenvectors
            opt = struct('issym', true, 'isreal', true);
            [U, ~] = eigs(L, D, k, 'sm', opt);
            
            % perform k-means
            Y = model.clustering.KmeansClustering.cluster(U, k);
        end
    end
    
end

