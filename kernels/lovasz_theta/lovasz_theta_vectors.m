%
% Computes an orthogonal vector labelling
% for the adjacency matrix A
% using Lovasz theta by SDP (CVX)
%
% Author: Fredrik Johansson
%
% A is an adjacency matrix of a graph
% d is the desired dimension of vector labelling
% If corrNonPSD is set to true, small errors in
%    psd-property  of X is corrected
% options: verbose
%
% U is a vector labelling
% c is a handle
% t is Lovasz theta number
% X is the SDP result

function [U,c,t,X] = lovasz_theta_vectors(A,d,corrNonPSD,opt);

if(nargin<2 || isempty(d))
    if(~iscell(A))
        d = size(A,1)+1;
    else
        maxd = 0;
        for i=1:length(A)
            maxd = max(maxd,size(A{i},1));
        end
        d = maxd+1;
    end
end
if(nargin<3)
    corrNonPSD = true;
end
if(nargin<4)
    opt.verbose = false;
end

if(~iscell(A))
    n = size(A,1);
    [X, t, dual_vars] = lovasz_theta_cvx(1-A-eye(n));
    [U c] = lovasz_theta_labelling(X,t,d,corrNonPSD);
else
    N = length(A);
    U = cell(N,1);
    X = cell(N,1);
    t = cell(N,1);
    c = cell(N,1);
    for i=1:N
        if(opt.verbose)
            disp(sprintf('Computing labelling %d/%d...',i,N));
        end
        n = size(A{i},1);
        [Xi, ti, dual_vars] = lovasz_theta_cvx(1-A{i}-eye(n));
        [Ui ci] = lovasz_theta_labelling(Xi,ti,d,corrNonPSD);
        U{i} = Ui; X{i} = Xi; t{i} = ti; c{i} = ci;
    end
end

