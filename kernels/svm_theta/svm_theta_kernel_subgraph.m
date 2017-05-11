%
% The original SVM theta kernel used in ICML paper
%
function [K, runtime, out] = svm_theta_kernel_subgraph(Gs, opt)

default_options = struct(...
    'verbose',              false, ...
    'subgraph_samples',     100, ...
    'sampling_weights',     'log', ...
    'kernel',               'linear', ...
    'sampling_scheme',      'convergence', ...
    'divide_by_d',          false, ...
    'laplacian',            false, ...
    'alpha_output',         false, ...
    'precomputed_alpha',    false, ...
    'tolerance',            10^-4, ...
    'sigma',                1 ...
);

opt = struct_union(default_options, opt);

N = length(Gs);

mind = 1;
min_samples = 5;

tstart = cputime;

% -- COMPUTE ALPHAS ---------------------------------------------
if(~opt.precomputed_alpha)
    alphas = {};
    ns = [];
    if(opt.verbose)
        fprintf(1,'Computing SVM-theta...\n');
    end
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
    alphas = opt.precomputed_alpha;
    ns = [];
    maxSumAlpha = 0;
    for i=1:N
        ns(i) = size(Gs{i},1);
        maxSumAlpha = max(sum(alphas{i}),maxSumAlpha);
    end
end

if(opt.alpha_output)
    save(opt.alpha_output,'alphas');
end

minn = min(ns);
maxd = max(ns);

nBins = maxd-mind+1;

nSamplesPer = zeros(nBins,N);
if(~strcmp(opt.sampling_scheme,'convergence'))
    nSamplesPer = sample_weights(opt.subgraph_samples, [mind, maxd], ns, opt.samplingWeights);
end

K = zeros(N,N);

avgAlphas = zeros(nBins,N);
mean_diffs = zeros(nBins,1);

sample_a = zeros(N,opt.subgraph_samples);
sample_d = zeros(N,opt.subgraph_samples);

% -- SAMPLE SUBSETS ----------------------------------------------
if(opt.verbose)
    fprintf(1,'Sampling subsets...\n');
end

for i=1:N
    if(opt.verbose)
        progresscount(i,1,N);
    end
    alpha = alphas{i};
    n = size(Gs{i},1);
    si = 1;
    maxdi = min(maxd,n);
    for d=mind:maxdi
        alpha_d = [];
        
        if(strcmp(opt.sampling_scheme,'convergence'))
            k = 0;
            mean_diff = inf;
            mean_last = inf;
            while(mean_diff > opt.tolerance && k<opt.subgraph_samples)
                k = k+1;
                P = randperm(n);
                p = P(1:d);        
                a =  sum(alpha(p));
                alpha_d(end+1) = a;
                
                sample_a(i,si) = a;
                sample_d(i,si) = d;
        
                si = si+1;
                if(k>min_samples)
                    mean_diff = sum((mean_last-mean(alpha_d)).^2)/sum(mean(alpha_d).^2);
                end
                mean_last = mean(alpha_d);
            end    
            nSamplesPer(d,i) = k;
            mean_diffs(d) = mean_diff;
        else
            for s=1:nSamplesPer(d-mind+1,i)
                P = randperm(n);
                p = P(1:d);        
                a =  sum(alpha(p));
                alpha_d(end+1) = a;
                
                sample_a(i,si) = a;
                sample_d(i,si) = d;
                
                si = si+1;
            end
        end
        avgAlphas(d-mind+1,i) = mean(alpha_d);
    end
end



% -- COMPUTE KERNEL ----------------------------------------------
if(opt.verbose)
    fprintf(1,'Computing SVM-theta kernel...\n');
end

for i=1:N
    if(opt.verbose)
        progresscount(i,1,N);
    end
    for j=i:N
        ds = mind:min([maxd ns(i) ns(j)]);
        if(strcmp(opt.kernel,'linear'))
            ads = mind:(size(avgAlphas)+mind-1);
            a1 = avgAlphas(:,i);
            if(opt.divide_by_d)
                a1 = a1./ads;
            end
            k = a1'*avgAlphas(:,j);
            
        else % -- Gaussian kernel 
            k = 0;
            for d=ds
                kd = 0;
                
                smpls_i = sample_a(i,sample_d(i,:)==d);
                smpls_j = sample_a(j,sample_d(j,:)==d);
                nSamples_i = length(smpls_i);
                nSamples_j = length(smpls_j);
                
                if(nSamples_i > 0 && nSamples_j > 0) 
                    R1 = smpls_i'*ones(1,nSamples_j);
                    R2 = ones(nSamples_i,1)*smpls_j;
                    R = rbf_kernel(R1,R2,opt.sigma);
                    kd = kd + sum(R(:));
                    kd = kd/(nSamples_i*nSamples_j);
                end     
                if(opt.divide_by_d)
                    kd = kd/d;
                end
                k = k+kd;
            end
        end
        
        K(i,j) = k;
        K(j,i) = k;
    end
end

out = struct();
out.alphas = alphas;
out.nSamplesPer = nSamplesPer;

runtime = cputime-tstart;
