classdef KernelMatrix < handle

    properties
        X, cachedMatrix, isCacheOn;
    end
    
    methods
        function km = KernelMatrix(X, isCacheOn)
            n = size(X, 1);
            km.X = X;
            km.cachedMatrix = sparse(n, n);
            km.isCacheOn = isCacheOn;
        end
        
        function value = get(obj, i, j)
            if (obj.isCacheOn == 0)
                value = model.classify.KernelMatrix.kernelFunction(obj.X(i, :), obj.X(j, :));
            else
                % speed up looking up
                if (i > j)
                    tmp = i;
                    i = j;
                    j = tmp;
                end
                
                if (obj.cachedMatrix(i, j) == 0)
                    obj.cachedMatrix(i, j) = model.classify.KernelMatrix.kernelFunction(obj.X(i, :), obj.X(j, :));
                end
                value = obj.cachedMatrix(i, j);
            end
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

