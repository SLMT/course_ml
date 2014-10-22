function drawResult( param, X, y )
    
    % draw line
    lineX = linspace(-10, 10);
    lineY = param(4) * lineX.^2.72727 + param(3) * lineX.^2 + param(2) * lineX + param(1);
    plot(lineX, lineY);
    hold on;
    
    % draw data set
    for t = 1 : size(X, 1)
        if y(t) == 1
            plot(X(t), 0, 'ro');
        else
            plot(X(t), 0, 'go');
        end
    end
    hold off;

end

