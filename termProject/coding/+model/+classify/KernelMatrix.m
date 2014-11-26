classdef KernelMatrix < handle

    properties
        X, cachedMatrix;
    end
    
    methods
        function km = KernelMatrix(X)
            n = size(X, 1);
            km.X = X;
            km.cachedMatrix = sparse(n, n);
        end
        
        function value = get(obj, i, j)
            if (obj.cachedMatrix(i, j) == 0)
                obj.cachedMatrix(i, j) = obj.kernelFunction(i, j);
            end
            value = obj.cachedMatrix(i, j);
        end
        
        function value = kernelFunction (obj, i, j)
            gama = 0.001;
            a = obj.X(i);
            b = obj.X(j);
            % value = exp(-gama * (norm(a - b, 2) ^ 2));
            value = a' * b;
        end
    end
    
end

