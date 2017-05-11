function [X, t, dual_vars] = lovasz_theta_cvx(A)

    n = size(A, 1); 
    [I, J] = find(A); 
    m = length(I); 
    
    cvx_clear;
    cvx_quiet(true);
    cvx_solver SDPT3
    cvx_begin sdp 
        variable X(n, n) ;
        variable t ; 
        dual variables Y W Z; 
        minimize t;
        subject to 
            for i=1:m
                ci = I(i); cj = J(i); 
                X(ci , cj )   == -1; 
            end
            Z: X == semidefinite(n) ;
            W: diag(X) == t - 1;
    cvx_end    
    dual_vars = struct('Y' , Y, 'Z', Z, 'W', W); 
end 