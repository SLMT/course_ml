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
                obj.cachedMatrix(i, j) = model.classify.KernelMatrix.kernelFunction(obj.X(i, :), obj.X(j, :));
            end
            value = obj.cachedMatrix(i, j);
        end
    end
    
    methods (Static)
        function value = kernelFunction (a, b)
            % Gaussian Radial Basis Function
            gama = 0.001;
            value = exp(-gama * (norm((a - b)', 2) ^ 2));
            
            % Linear Kernel
            % value = a * b';
        end
    end
    
end

