function [c,r] = fitball(A,opt)

d = size(A,1);
n = size(A,2);


if(n==0)
    c = zeros(d,1);
    r = 0;
elseif(n==1)
    c = A(:,1);
    r = 0;
else
    Q = A-A(:,1)*ones(n,1)';
   
    B = 2*(Q'*Q);
    b = diag(B)/2;
    
%     L = linsolve(B(2:end,2:end),b(2:end,:),opt);
    L = mldivide(B(2:end,2:end),b(2:end,:));
    
    L = [0; L];
    C = zeros(d,1);
    for i=2:n
    C = C+  L(i)*Q(:,i);
    end

    r = sqrt(C'*C);
    c = C+A(:,1);   
end
