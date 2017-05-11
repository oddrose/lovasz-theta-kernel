%
% Implementation of original svmtheta kernel 
% using the convolution kernel function.
%
function [K, runtime, alphas] = svm_theta_kernel(Gs,nSamples,opt)

if(nargin<2)
    nSamples = 20;
end
if(nargin<3)
    opt = struct();
end

opt = process_option_struct(opt,...
    {'verbose','samplingWeights','kernel','samplingScheme',...
    'divideByD','laplacian','alphaOutput','precomputedAlpha','tolerance','sigma'},...
    {false,'log','linear','convergence',false,false,false,{},10^-4,1});

N = length(Gs);

mind = 2;
minSamples = 3;

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

minn = min(ns);
maxd = max(ns);

nBins = maxd-mind+1;

nSamplesPer = zeros(nBins,N);
if(~strcmp(opt.samplingScheme,'convergence'))
    nSamplesPer = sample_weights(nSamples, [mind, maxd], ns, opt.samplingWeights);
end

phis = [];    
avgAlphas = [];

convopt = struct();
convopt.nSamples = nSamples;
convopt.tolerance = 1e-6;
convopt.nMinSamples = 20;
convopt.divideByD = true;
% kfun = @(i,j,pi,pj) svm_theta_conv(i,j,pi,pj,alphas);

fprintf(1,'Computing SVM-theta kernel...\n');
K = convolution_kernel(Gs,@svm_theta_kfun,...
    @svm_theta_statfun,alphas,convopt);


function k = svm_theta_kfun(alphas,pi,pj)
    k = sum(alphas{1}(pi))*sum(alphas{2}(pj));

function s = svm_theta_statfun(alphas,pi)
    s = sum(alphas(pi));

runtime = cputime-tstart;
