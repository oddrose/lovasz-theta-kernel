%
% Compute the Lovasz-theta and SVM-theta kernels 
% on the MUTAG dataset.
%
% Author: Fredrik Johansson, 2012-2017
%

clearvars;
close all;

addpath(genpath('./'));

% -- Add necessary paths
setenv('LIBSVMHOME','./lib/libsvm');

% -- Define options
opt = struct(...
    'verbose',      true, ...
    'output_dir',   './results/', ...
    'dataset',      'MUTAG', ...
    'precomputed',  false, ...
    'svm_theta_variant', 'subgraph-label' ...
);

% -- Create output folder
if ~isdir(opt.output_dir) 
    mkdir(opt.output_dir);
end

% -- Load datafile
datafile = 'data/MUTAG.mat';
load(datafile);
graphs = MUTAG; 
labels = lmutag;
N = length(graphs);
clear MUTAG lmutag;

%% Compute Lovasz kernel
[K_lo, t_lo] = lovasz_theta_kernel(graphs, opt);

%% Compute SVM-theta kernel
[K_svm, t_svm] = svm_theta_kernel(graphs, opt);

%%

opt = struct();

opt.kernelsUsed = {'svm'};
opt.verbose = true;

opt.wlopt = struct();
opt.wlopt.iterations = 3;

opt.svmopt = struct();
opt.svmopt.variant = 'matching';
opt.svmopt.kernel = 'linear'; %gaussian / linear
opt.svmopt.samples = 100;
opt.svmopt.tolerance = 1e-4;
opt.svmopt.sigma = 0.1;
opt.svmopt.verbose = true;
opt.svmopt.alpha = 1;
opt.svmopt.addDiag = false;
opt.svmopt.embeddingType = 'laplacian';
opt.svmopt.wlIterations = 3;
opt.svmopt.lsLabellingFile = sprintf('%s/%s_ls_labellings.mat',resultDir,dataset);
opt.svmopt.embeddings = Us;

removeNodeLabels = false;

data = {};
for i=1:length(graphs)
    data{i} = graphs(i).am;
    if(removeNodeLabels)
        graphs(i).nl.values = graphs(i).nl.values*0+1;
    end
end

% opt.fraikinKernelFile = [result_dir,'/MUTAG_lovasz_fraikin.mat'];
% opt.graphletKernelFile = [result_dir,'/MUTAG_graphlet_nino.mat'];
% opt.shortestPathKernelFile = [result_dir,'/MUTAG_shortest_path_nino.mat'];
% opt.randomWalkKernelFile = [result_dir,'/MUTAG_randomwalk.mat'];
% opt.svmThetaKernelFile = [result_dir,'/MUTAG_svmtheta.mat'];
% opt.svmThetaAlphaFile = [result_dir,'/MUTAG_svmtheta_alpha.mat'];
% opt.wlKernelFile = [result_dir,'/MUTAG_weisfeiler-lehman.mat'];
opt.graphs = graphs;

[R runtimes stat] = test_kernels(data,labels,'MUTAG',resultDir,opt);

for i=1:length(R)
    fprintf(1,'\n%s\n',R(i).label);
    fprintf(1,'Best CV: %.3f%%, c=%.3f, g=%.3f\n',R(i).cv,R(i).c,R(i).g);
end
