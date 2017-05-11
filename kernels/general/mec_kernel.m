function [K, runtime] = mec_kernel(U, opt)

default_options = struct(...
    'output_dir',       false, ...
    'dataset',          'unspecified', ...
    'embeddings',       false, ...
    'mec_samples',      50, ...
    'mec_sigma',        0.1, ...
    'max_subset_size',  8, ...
    'min_subset_size',  2, ...
    'kernel',           'gaussian', ...
    'use_angle',        false, ...
    'uni_samples',      false, ...
    'always_max_d',     true, ...
    'n_min_samples',    0, ...
    'theta',            false ...
);

% -- Merge option structs (override defaults if supplied)
opt = struct_union(default_options, opt);

% --- Compute smallest number of points in any embedding
N = length(U);
maxn = 0;
for i=1:N
    maxn = max(maxn,size(U{i},2));
end

% --- Sample a bunch of subsets for each labelling
A = zeros(N, opt.mec_samples);
D = zeros(N, opt.mec_samples);
f_avg = zeros(N, min(maxn, opt.max_subset_size));

tstart = cputime;
if(opt.verbose)
    fprintf(1,'Sampling and computing angles...\n');
end
for i=1:N
    if(opt.verbose)
        progresscount(i,1,N);
    end
    u = U{i};
    n = size(u, 2);
    
    maxd = min(opt.max_subset_size, n);
    w = ones(maxd, 1);
    if(~opt.uni_samples)
        for d=1:maxd
            w(d) = nchoosek(n, d);
        end
    end
    w = w/sum(w);
    
    smpls = floor(w * opt.mec_samples);
        
    for r = 1:(opt.mec_samples - sum(smpls))
        ri = mod(r-1,maxd)+1;
        smpls(ri) = smpls(ri)+1;
    end
    for d = 1:length(smpls)
        smpls(d) = max(smpls(d), opt.n_min_samples);
    end
    
    ds = opt.min_subset_size:maxd;
    if(maxd < n && opt.always_max_d)
        ds = [ds n];
    end

    si = 1;
    for d = ds
        S = [];
        if(d<n)
            for k = 1:smpls(d)
                P = randperm(n);
                p = P(1:d);
                v = u(:,p);

                if(d>1)
                    t = minimum_cone(v);
                    if(~opt.use_angle)
                        t = cos(t);
                    end
                else
                    t = 0;
                end

                if(~isreal(t))
                    error('Imaginary angle, t');
                end

                A(i,si) = t;
                D(i,si) = d;
                si = si+1;
                S(k) = t;
            end
        elseif(d==n)
            if(opt.theta)
                t = opt.theta(i);
            else
                t = minimum_cone(u);
            end
            if(~opt.use_angle)
                t = cos(t);
            end
            A(i,si) = t;
            D(i,si) = d;
            si = si+1;
            S = t;
        end
        if(sum(S) ~= 0)
            f_avg(i,d) = mean(S);
        else
            f_avg(i,d) = 0;
        end
    end
end

% --- Compute Kernel matrix
K = zeros(N, N);
for i=1:N
    if(opt.verbose)
        fprintf(1,'Computing kernel row for set %d/%d\n', i, N);
    end
    for j=i:N
        k = 0;    
        if(strcmp(opt.kernel, 'linear'))
            k = f_avg(i, :) * f_avg(j, :)';
        else
            nmin = min(size(U{i}, 2), size(U{j}, 2));
            maxd = min(nmin, opt.max_subset_size);
            ds = opt.min_subset_size:maxd;
            if(maxd < nmin && opt.always_max_d)
                ds = [ds nmin];
            end

            for d=ds
                kd = 0;
                smpls_i = A(i,D(i,:)==d);
                smpls_j = A(j,D(j,:)==d);
                n_samples_i = length(smpls_i);
                n_samples_j = length(smpls_j);
                if(n_samples_i > 0 && n_samples_j > 0) 
                    R1 = smpls_i'*ones(1, n_samples_j);
                    R2 = ones(n_samples_i, 1)*smpls_j;
                    R = rbf_kernel(R1, R2, opt.mec_sigma);
                    kd = kd + sum(R(:));
                    k = k + kd / (n_samples_i * n_samples_j);
                end            
            end
        end
        K(i,j) = k;
        K(j,i) = k;
    end
end
runtime = cputime-tstart;