function [alpha K_LS] = svm_theta_alpha(A,C,nl,rho,useLaplacian)

n = size(A,1);
if(nargin<2)
    C = n;
end
if(nargin<3)
    nl = ones(n,1);
end

if(nargin>2)
    A = A.*sqrt(nl*nl');
end

if(nargin<5)
    useLaplacian = false;
end

if(nargin<4 || isempty(rho))
    if(~useLaplacian)
        rho = -min(eig(A))+0.00001;
    else
        rho = 1;
    end
end

% keyboard
if(useLaplacian)
    K = graph_laplacian(A,true);
else
    K = A/rho + eye(size(A,1));
end
K_LS = K;

labels = [ones(n,1); -1];

SVMROOT = getenv('LIBSVMHOME');
kernelFile = 'svm-theta-kernel.tmp';
modelFile = 'svm-theta-model.tmp';

K = [2*K zeros(n,1); zeros(1,n) 0];
svmWriteKernel(K, kernelFile, labels); 
unix(sprintf('%s/svm-train -q -s 0 -c %g -t 4 %s %s', SVMROOT, C, kernelFile, modelFile));
model = svmReadModel(modelFile);

delete(kernelFile,modelFile);

alpha1 = model.sv_coef; alpha = zeros(n,1);
alpha(model.SVs(1:end-1)) = alpha1(1:end-1);
