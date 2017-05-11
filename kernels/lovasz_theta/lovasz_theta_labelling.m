function [U c] = lovasz_theta_labelling(X,t,d,corrNonPSD)

n = size(X,1);

if(nargin<3)
    d = n+1;
else
    if(d<n+1)
        fprintf(2,'WARNING: supplied d < n+1, d=%d,n=%d\n',d,n);
    end
end

if(nargin<4)
    corrNonPSD = true;
end


try
    V = chol(X);
catch error
    if(corrNonPSD)
        Y = X + 2*abs(min(eig(X)))*eye(n);
        V = chol(Y);
    else
        rethrow(error);
    end
end

V = [V; zeros(d-n,n)];
c = [zeros(d-1,1); 1];

C = c*ones(n,1)';

U = 1/sqrt(t)*(C+V);