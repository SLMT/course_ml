function [ normalizedX ] = zNormalize( X, Xmean, Xstd )

n = size(X, 1);

normalizedX = (X - repmat(Xmean,[n,1])) ./ repmat(Xstd,[n,1]); %Repmat for Dimension agreeement

end
