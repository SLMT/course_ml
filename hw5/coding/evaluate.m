function result = evaluate( X, Y )
    
    % initialize variables
    m = size(X, 1);
    d = size(X, 2);
    k = size(Y, 2);

    % calculate means for each cluster
    means = zeros(k, d);
    for i = 1 : k
        means(i, :) = mean(X(Y(:, i) == 1, :));
    end
    
    % calculate intercluster separation
    inter = 0;
    for i = 1 : k
        for j = 1 : k
            inter = inter + norm(means(i) - means(j)) ^ 2;
        end
    end
    
    % calculate intracluster cohesion
    intra = 0;
    for i = 1 : k
        data = X(Y(:, i) == 1, :);
        num = size(data, 1);
        
        sum = 0;
        for j = 1 : num
            sum = sum + norm(data(j, :) - means(i, :)) ^ 2;
        end
        
        intra = intra + sum / num;
    end
    
    % result
    result = inter / intra;
end

