function K = lovasz_theta_kernel(graphs, opt)

default_options = struct(...
    'output_dir',       false, ...
    'dataset',          'unspecified', ...
    'precomputed',       false ...
);

% -- Merge option structs (override defaults if supplied)
opt = struct_union(default_options, opt);

% -- Fetch or compute embeddings
embedding_file = sprintf('%s/%s_lovasz_embeddings.mat', opt.output_dir, opt.dataset);
if opt.precomputed == false
    % -- Compute Lovasz embedding (and store if specified)
    if opt.output_dir ~= false
        Ls = lovasz_theta_embedding(graphs,embedding_file);
    end
else
    load(embedding_file);
end

% -- Compute Minimum Enclosing Cone kernel
U = {Ls.U};
K = mec_kernel(U, opt);