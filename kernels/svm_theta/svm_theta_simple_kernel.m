function [K  runtime out] = svm_theta_simple_kernel(Gs,opt)

if(nargin<2)
    nSamples = 20;
end
if(nargin<2)
    opt = struct();
    opt.verbose = false;
end
if(~isfield(opt,'verbose'))
    opt.verbose = false;
end
if(~isfield(opt,'sort_features'))
    opt.sort_features = false;
end
if(~isfield(opt,'nl'))
    useLabels = false;
else
    useLabels = true;
end

N = length(Gs);

tstart = cputime;

% -- COMPUTE ALPHAS ---------------------------------------------
alphas = {};
fprintf(1,'Computing SVM-theta...\n');

for i=1:N
    if(opt.verbose)
        progresscount(i,1,N);
    end
    if(useLabels)
        alphas{i} = svm_theta_alpha(Gs{i},size(Gs{i},1),opt.nl(i).values);
    else
        alphas{i} = svm_theta_alpha(Gs{i},size(Gs{i},1));
    end
    
end


% -- COMPUTE KERNEL ----------------------------------------------
K = zeros(N,N);
fprintf(1,'Computing SVM-theta Simple kernel...\n');
for i=1:N
    if(opt.verbose)
        progresscount(i,1,N);
    end
    for j=i:N
        di = sort(alphas{i},'descend');
        dj = sort(alphas{j},'descend');
        minn = min(size(Gs{i},1),size(Gs{j},1));
        k = di(1:minn)'*dj(1:minn);
        
        ni = size(Gs{i},1); nj = size(Gs{j},1);
        maxn = max(ni,nj);
        di = [alphas{i}; zeros(max(maxn-ni,0),1)];
        dj = [alphas{j}; zeros(max(maxn-nj,0),1)];
        
        if(opt.sort_features)
            di = sort(di,'descend');
            dj = sort(dj,'descend');
        end
        
        k = di'*dj;
        
        K(i,j) = k;
        K(j,i) = k;
    end
end

runtime = cputime-tstart;
out = struct();
out.alphas = alphas;