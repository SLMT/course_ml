classdef SMOClassifier < handle

    properties
        alpha, w, b;
    end
    
    methods
        function smoClassifierObj = SMOClassifier(alpha, w, b)
            smoClassifierObj.alpha = alpha;
            smoClassifierObj.w = w;
            smoClassifierObj.b = b;
        end
        function predictedLabel = predict (obj, X)
            
        end
    end
    
    methods (Static)
        function smoClassifierObj = train (X, y)
            % some constants
            epsilon = 0.00001;
        	n = size(X, 1);
            d = size(X, 2);
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
                if ygMatrix(i) < ygMatrix(j) + epsilon
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
            
            % calculate w, b
            w = zeros(d, 1);
            for t = 1 : n
                if (a(t) > epsilon)
                    w = w + a(t) * y(t) * X(t, :)';
                end
            end
            
            b = 0;
            num = 0;
            for t = 1 : n
                if (a(t) > epsilon && a(t) < c - epsilon)
                    b = b + y(t) - X(t, :) * w;
                    num = num + 1;
                end
            end
            b = b / num;
            smoClassifierObj = model.classify.SMOClassifier(a, w, b);
        end
	end
end

