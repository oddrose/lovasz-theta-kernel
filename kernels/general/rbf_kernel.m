function k = rbf_kernel(x,y,s)

if(size(x,1)~=size(y,1) || size(x,2) ~=size(y,2))
    error('input of different sizes');
end
if(size(x,1)==1 || size(x,2)==1)
    k = exp(-norm(x-y,2)^2/(2*s));
else
    % matrix - rbf on every element separately
    A = (x-y);
    A = A.^2;
    A = -A/(2*s);
    k = exp(A);
end