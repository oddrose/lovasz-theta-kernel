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
