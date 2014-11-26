classdef SMOClassifier < handle

    properties
        vecA;
    end
    
    methods
        function smoClassifierObj = SMOClassifier(vecA)
            smoClassifierObj.vecA = vecA;
        end
        function predictedLabel = predict (obj, X)
            % TODO: Fill this method
        end
    end
    
    methods (Static)
        function smoClassifierObj = train (X, y)
        	n = size(X, 1);
            c = 1 / n;
            km = model.classify.KernelMatrix(X);

            % SMO
            a = zeros(n, 1);
            g = ones(n, 1);

            % create constraints
            A = zeros(n, 1);
            B = zeros(n, 1);
            for t = 1 : n
                if y(t) == 1
                    B(t) = c;
                else
                    A(t) = -c;
                end
            end
            
            iteration = 0;
            while 1
                % pre-generate matrix
                ygMatrix = y .* g;
                yaMatrix = y .* a;
                
                % i = argmax(yg) subject to ya < B
                tmpIndex = yaMatrix < B;
                [~, maxI] = max(ygMatrix(tmpIndex));
                tmp = find(tmpIndex);
                i = tmp(maxI);
                
                % j = argmin(yg) subject to A < ya
                tmpIndex = yaMatrix > A;
                [~, minI] = min(ygMatrix(tmpIndex));
                tmp = find(tmpIndex);
                j = tmp(minI);
                
                % yigi < yjgj
                if ygMatrix(i) < ygMatrix(j) + 0.00001
                    break;
                end
                
                % Direction search
                candidate1 = B(i) - yaMatrix(i);
                candidate2 = yaMatrix(j) - A(j);
                candidate3 = (ygMatrix(i) - ygMatrix(j)) / (km.get(i,i) + km.get(j, j) - 2 * km.get(i, j));
                lamda = min([candidate1; candidate2; candidate3]);
                
                % update gradient
                for k = 1 : n
                    tmp = lamda * y(k);
                    g(k) = g(k) - tmp * km.get(i, k) + tmp * km.get(j, k);
                end
                
                % update coefficients
                a(i) = a(i) + y(i) * lamda;
                a(j) = a(j) - y(j) * lamda;
                
                iteration = iteration + 1;
            end

            iteration
            smoClassifierObj = model.classify.SMOClassifier(a);
        end
	end
end

