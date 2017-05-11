% Based on Welzl, 1991

function [c, r] = minidisk(A,opt)

if(nargin<2)
    opt = struct();
end
if(~isfield(opt,'SYM'))
    opt.SYM = true;
end

n = size(A,2);
P = randperm(n);

[c,r] = b_minidisk(A,P,[],opt);

