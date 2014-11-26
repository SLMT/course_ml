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
            y = X * obj.w - obj.b;
            predictedLabel = sign(y);
        end
    end
    
    methods (Static)
        function smoClassifierObj = train (X, y)
            % some constants
            epsilon = 0.00001;
            n = size(X, 1);
            d = size(X, 2);
            c = 1;
            km = model.classify.KernelMatrix(X);
            
            % =====================
            % ======== SMO ========
            % =====================
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
                if ygMatrix(i) <= ygMatrix(j)
                    break;
                end
                
                % Direction search
                candidate1 = B(i) - yaMatrix(i);
                candidate2 = yaMatrix(j) - A(j);
                candidate3 = (ygMatrix(i) - ygMatrix(j)) / (km.get(i,i) + km.get(j, j) - (2 * km.get(i, j)));
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
            
            % =====================
            % ==== End of SMO =====
            % =====================
            
            % calculate w, b
            optW = zeros(d, 1);
            for t = 1 : n
                if (a(t) > epsilon)
                    optW = optW + a(t) * y(t) * X(t, :)';
                end
            end
            
            optB = 0;
            num = 0;
            for t = 1 : n
                if (a(t) > epsilon && a(t) < c - epsilon)
                    optB = optB + y(t) - X(t, :) * optW;
                    num = num + 1;
                end
            end
            optB = optB / num;
            
            smoClassifierObj = model.classify.SMOClassifier(a, optW, optB);
        end
	end
end

