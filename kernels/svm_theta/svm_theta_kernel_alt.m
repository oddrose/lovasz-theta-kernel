function [K, runtime, out] = svm_theta_kernel_alt(Gs,nSamples,opt)

if(nargin<2)
    nSamples = 20;
end
if(nargin<3)
    opt = struct();
end

opt = process_option_struct(opt,...
    {'verbose','samplingWeights','kernel','samplingScheme',...
    'laplacian','alphaOutput','precomputedAlpha',...
    'tolerance','sigma'},...
    {false,'log','linear','convergence',...
    false,false,{},10^-4,1});

N = length(Gs);


tstart = cputime;

% -- COMPUTE ALPHAS ---------------------------------------------
if(isempty(opt.precomputedAlpha))
    alphas = {};
    ns = [];
    fprintf(1,'Computing SVM-theta...\n');
    maxSumAlpha = 0;
    for i=1:N
        if(opt.verbose)
            progresscount(i,1,N);
        end
        if(opt.laplacian) 
            A = graph_laplacian(Gs{i},false);
        else
            A = Gs{i};
        end
        alphas{i} = svm_theta_alpha(A);
        ns(i) = size(Gs{i},1);
        maxSumAlpha = max(sum(alphas{i}),maxSumAlpha);
    end
else
    alphas = opt.precomputedAlpha;
    ns = [];
    maxSumAlpha = 0;
    for i=1:N
        ns(i) = size(Gs{i},1);
        maxSumAlpha = max(sum(alphas{i}),maxSumAlpha);
    end
end

if(opt.alphaOutput)
    save(opt.alphaOutput,'alphas');
end


% -- SAMPLE SUBSETS ----------------------------------------------
fprintf(1,'Sampling subsets...\n');

avgAlphas = zeros(1,N);

sample_a = zeros(N,nSamples);
sample_d = zeros(N,nSamples);

for i=1:N
    if(opt.verbose)
        progresscount(i,1,N);
    end
    alpha = alphas{i};
    n = size(Gs{i},1);
    
    si = 1;
    for s=1:nSamples
        d = randi(n-1)+1;
        P = randperm(n);
        p = P(1:d);     
        
        a =  sum(alpha(p));
        
        sample_a(i,si) = a;
        sample_d(i,si) = d;
        
        si=si+1;
    end
    avgAlphas(:,i) = mean(sample_a(i,:));
end



% -- COMPUTE KERNEL ----------------------------------------------
K = zeros(N,N);
fprintf(1,'Computing SVM-theta kernel...\n');

for i=1:N
    if(opt.verbose)
        progresscount(i,1,N);
    end
    for j=i:N
        if(strcmp(opt.kernel,'linear'))
            a1 = avgAlphas(:,i);
            k = a1'*avgAlphas(:,j);
            
        else % -- Gaussian kernel 
            k = 0;
                
            smpls_i = sample_a(i,:);
            smpls_j = sample_a(j,:);
            nSamples_i = length(smpls_i);
            nSamples_j = length(smpls_j);

            if(nSamples_i > 0 && nSamples_j > 0) 
                R1 = smpls_i'*ones(1,nSamples_j);
                R2 = ones(nSamples_i,1)*smpls_j;
                R = rbf_kernel(R1,R2,opt.sigma);
                kd = sum(sum(R));
                kd = kd/(nSamples_i*nSamples_j);
            end     
            k = k+kd;
        end
        K(i,j) = k;
        K(j,i) = k;
    end
end

out = struct();
out.alphas = alphas;

runtime = cputime-tstart;
