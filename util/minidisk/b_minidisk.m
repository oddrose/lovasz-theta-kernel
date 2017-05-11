function [c,r] = b_minidisk(A,P,R,opt)

% ---- WELZL's ALGORITHM (MOVE-TO-FRONT)
d = size(A,1);

nP = length(P);
nR = length(R);
   

% -- ORIGINAL ALGORITHM
if(nP==0 || nR==d+1)   
	[c,r] = fitball(A(:,R),opt);
else
    p = P(randi(nP));
    P1 = P; P1(P1==p) = [];
    [c,r] = b_minidisk(A,P1,R,opt);
    if(~insideball(A(:,p),c,r))
        R1 = R;
        if(isempty(find(R==p,1,'first')))
            R1 = [R1, p];
            [c,r] = b_minidisk(A,P1,R1,opt);   
        end
    end
end