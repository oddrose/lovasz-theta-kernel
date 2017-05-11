function [t c] = minimum_cone(A)

%MINIMUM_CONE Computes the angle and center of the
%   minimum cone, with its point in the origin, 
%   enclosing the vectors in A.
%
%   [T C] = MINIMUM_CONE(A) returns the the angle T
%   and the center vector C of the minimum cone
%   enclosing by the set of vectors in A, each row
%   a coordinate and each column a vector.
%
%   Copyright (c) 2013, Fredrik Johansson 
%   frejohk@chalmers.se

N = size(A,2);
n = size(A,1);

opt = struct();
opt.SYM = true;
    
[c r] = minidisk(A,opt);
% [c r] = pivot_mb(A); % still not entirely correct

% Cube instead of ball:
% c = (min(A,[],2)+max(A,[],2))/2;
% norm(c2-c)

c = c/norm(c,2);

t = min(A'*c);
if(t>1 && t<1+10^-6)
    t = 1;
end
if(t<-1 && t>-1-10^-6)
    t = -1;
end
t = acos(t);
