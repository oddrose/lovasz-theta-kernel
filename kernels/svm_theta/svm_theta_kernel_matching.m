function [K runtime out] = svm_theta_kernel_matching(Gs,opt)

opt = process_option_struct(opt,...
    {'alpha','verbose','addDiag','embeddingType','wlIterations'},...
    {0,false,true,'adjacency',0});

N = length(Gs);

tstart = cputime;

dmax = 0;
nmax = 0;
LSLs = cell(N,1);
for i=1:N
    dmax = max(dmax,size(Gs(i).am,1));
    nmax = max(nmax,size(Gs(i).am,2));
end

if(~isfield(opt,'embeddings'))
    for i=1:N
        if(strcmp(opt.embeddingType,'laplacian'))
            try 
                L = graph_laplacian(full(Gs(i).am),false);
                LSLs{i} = chol(L+0.001*eye(size(Gs(i).am)));
            catch
                keyboard
            end
        else
            LSLs{i} = ls_labelling(Gs(i).am);
        end
    end
    if(isfield(opt,'lsLabellingFile'))
       save(opt.lsLabellingFile,'LSLs');
    end
else
    LSLs = opt.embeddings;
end

LSpad = cell(N,1);
for i=1:N
    LS = LSLs{i};
    d = size(LS,1);
    n = size(LS,2);
    LSpad{i} = [LS(1:end-1,:); zeros(dmax-d,n); LS(end,:)];
end

Lh = {};
if(opt.wlIterations>0)
   Lh = WLlabels(Gs,opt.wlIterations,1,opt.verbose);
end

% -- Indefinite kernel with LS
K = zeros(N,N);
for i=1:N
    if(opt.verbose)
        progresscount(i,1,N);
    end
    ni = size(Gs(i).am,1);  
    for j=i:N
        nj = size(Gs(j).am,1);
        
        nli = Gs(i).nl.values;
        Nli = nli*ones(nj,1)';
        nlj = Gs(j).nl.values;
        Nlj = ones(ni,1)*nlj';
        B = Nli==Nlj;
        
        % Cosine similarity (assuming unit vectors)
        A = 1-pdist2(LSpad{i}',LSpad{j}','cosine');
        
        % Match based on node labels
%       C = (1-opt.alpha)*A+opt.alpha*B;

        C = A;
        [a b c] = bipartite_matching(C);
        
%         [a b c] = blossom_matching(A); % slow because of read/write
        
        nmatch = sum(Gs(i).nl.values(b)==Gs(j).nl.values(c));
        K(i,j) = a*(1-opt.alpha + opt.alpha*nmatch/min(size(C)));
            
        % Weight by fraction of correct label matches
        
        for h=1:opt.wlIterations
            nmatch = sum(Lh{h+1}{i}(b)==Lh{h+1}{j}(c));
            K(i,j) = K(i,j) + a*(1-opt.alpha + opt.alpha*nmatch/min(size(C)));
        end
        
        K(j,i) = K(i,j);
    end
end

if(opt.addDiag)
    K = K+eye(N)*(0.001-min(eig(K)));
end

runtime = cputime-tstart;
out = struct();