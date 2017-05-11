%
% General SVM theta kernel
%
function [K, runtime, out] = svm_theta_kernel(graphs,opt)

default_options = struct(...
    'svm_theta_variant', 'subgraph' ...
);

opt = struct_union(default_options, opt);

data = {};
for i=1:length(graphs)
    data{i} = graphs(i).am;
end

if(strcmp(opt.svm_theta_variant, 'subgraph'))
   [K, runtime, out] = ...
       svm_theta_kernel_subgraph(data, opt); 
   
elseif(strcmp(opt.svm_theta_variant, 'subgraph-label'))
   [K, runtime, out] = ...
       svm_theta_kernel_subgraph_l(graphs, opt); 
   
% TODO: Need to fix input structure in these
% elseif(strcmp(opt.svm_theta_variant, 'matching'))
%    [K, runtime, out] = ...
%        svm_theta_kernel_matching(graphs, opt); 
%    
% elseif(strcmp(opt.svm_theta_variant, 'alt'))
%    [K, runtime, out] = ...
%        svm_theta_kernel_alt(data, opt);
%    
% elseif(strcmp(opt.svm_theta_variant, 'simple'))
%     opt.nl = [graphs.nl];
%     for i=1:length(graphs)
%         opt.nl(i).values = opt.nl(i).values+1;
%     end
%    [K, runtime, out] = ...
%        svm_theta_simple_kernel(data, opt);
   
else
    
    error(sprintf('Unknown variant of SVM theta kernel: %s',opt.svm_theta_variant));
end
