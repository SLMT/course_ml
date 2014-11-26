classdef SoftMarginLinearClassifier < handle

    properties
        w, b, xi;
    end
    
    methods
        function smlClassifierObj = SoftMarginLinearClassifier(w, b, xi)
            smlClassifierObj.w = w;
            smlClassifierObj.b = b;
            smlClassifierObj.xi = xi;
        end
        function predictedLabel = predict (obj, X)
            y = X * obj.w - obj.b;
            predictedLabel = sign(y);
        end
    end
    
    methods (Static)
        function smlClassifierObj = train (X, y)
            n = size(X, 1);
            d = size(X, 2);
            c = 0.1;

            % Prepare
            C = zeros(n, 1);
            C(:) = c;

            % CVX
            cvx_begin
                variable varW(d)
                variable varB
                variable varXi(n) nonnegative
                minimize( norm(varW, 2) + C' * varXi )
                subject to
                    y .* (varW' * X' - varB)' > 1 - varXi
            cvx_end

            smlClassifierObj = model.classify.SoftMarginLinearClassifier(varW, varB, varXi);
        end
	end
end

