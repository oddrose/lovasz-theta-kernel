function b = insideball(p,c,r,opt)

tol = 1e-1;

if(norm(p-c,2)-r<=tol)
    b = true;
else
    b = false;
end