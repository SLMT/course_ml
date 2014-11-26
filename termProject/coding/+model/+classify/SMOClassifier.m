classdef SMOClassifier < handle

    properties
        alpha, data, labels;
    end
    
    methods
        function smoClassifierObj = SMOClassifier(alpha, data, labels)
            smoClassifierObj.alpha = alpha;
            smoClassifierObj.data = data;
            smoClassifierObj.labels = labels;
        end
        function predictedLabel = predict (obj, X)
            dataN = size(obj.data, 1);
            xN = size(X, 1);
            
            predictedLabel = zeros(xN, 1);
            for t = 1 : xN
                sum = 0;
                for i = 1 : dataN
                    sum = obj.labels(i) * obj.alpha(i) * model.classify.KernelMatrix.kernelFunction(obj.data(i, :), X(t, :));
                end
                predictedLabel(t) = sign(sum);
            end
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
            svIndex = a > epsilon;
            svAlpha = a(svIndex);
            svData = X(svIndex, :);
            svLabels = y(svIndex);

            smoClassifierObj = model.classify.SMOClassifier(svAlpha, svData, svLabels);
        end
	end
end

