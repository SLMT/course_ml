function plot_result( X, y )

    plot (X(y==1,1),X(y==1,2),'go');
    hold on;
    plot (X(y==-1,1),X(y==-1,2),'bo');
    hold on;
    plot (X(y==0,1),X(y==0,2), 'k.');
    hold off;

end

