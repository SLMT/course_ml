
y = [1; -1; 1; -1; -1];

a = [1; 2; 3; 4; 5];
g = [1; 2; 3; 4; 5];

I = (y .* a) < 0;
tmp = y .* g;
[C i] = max(tmp(I));
tmp = find(I);
tmp(i)