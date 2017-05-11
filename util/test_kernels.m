
function [R runtimes stat Kout] = test_kernels(data,labels,datasetName,outputFolder,opt)

opt = process_option_struct(opt,...
    {'classify','kernelsUsed','verbose','outputMat'...
     'sampleGraphlets',...
     'svmopt','wlopt','glopt',...
     'mecSamples'},...
     {true, {'all'}, false, true,...
     false, ...
     struct(), struct(), struct(),...
     100});

useAll = sum(ismember(opt.kernelsUsed,'all'))>0;

runtimes = struct();
stat = struct();
Kout = struct();

N = length(data);

if(opt.verbose)
    print_exp_header(opt.kernelsUsed);
end

% ------------ GRAPHLET KERNEL --------------------

if(useAll || sum(ismember(opt.kernelsUsed,'gl')))
    if(~isfield(opt,'graphletKernelFile'))
        
        opt.glopt = process_option_struct(opt.glopt,...
            {'delta','epsilon','graphletSizes','verbose'},...
            {0.05,0.05,[4],false});
        
        if(opt.verbose)
            fprintf(1,'Computing graphlet kernel...\n');
        end
        if(opt.sampleGraphlets)
            
            delta = opt.glopt.delta; epsilon = opt.glopt.epsilon;
            d = samplesize(delta,epsilon,11);
            Gs = {};
            for i=1:N
                Gs{i} = sparse(data{i});
            end
            [K_gl, freq, runtime] = graphletKernelMatrix(Gs,delta,epsilon,opt.glopt);
        else
            [K_gl, runtime,freq] = allkernel(opt.graphs,4);            
        end
        stat.gl = struct();
        stat.gl.freq = freq;

        Kout.gl = K_gl;
        outname = sprintf('%s/%s_graphlet.mat',outputFolder,datasetName);    
        runtimes.gl = runtime;
        save(outname,'K_gl','runtime');
    else
        load(opt.graphletKernelFile);
    end
end


% ------------ SVM-THETA KERNEL --------------------

if(useAll || sum(ismember(opt.kernelsUsed,'svm')))
    if(~isfield(opt,'svmThetaKernelFile'))
        % -- Compute SVM-Theta Kernel
        

        alphaoutname = sprintf('%s/%s_svmtheta_alpha.mat',outputFolder,datasetName);
        opt.svmopt.alphaOutput = alphaoutname;
        if(isfield(opt,'embeddings'))
            opt.svmopt.embeddings = opt.embeddings;
        end
        
        [K_svm runtime svmout] = svm_theta_kernel(opt.graphs,opt.svmopt);
            
        stat.svm = svmout;
        runtimes.svm = runtime;

        Kout.svm = K_svm;
        outname = sprintf('%s/%s_svmtheta.mat',outputFolder,datasetName);
        save(outname,'K_svm','runtime');%,'K_svm_s','K_svm_sl');
    else
        load(opt.svmThetaKernelFile);
    end
end

% ------------ RANDOM WALK KERNEL --------------------

if(useAll || sum(ismember(opt.kernelsUsed,'rw')))
    if(~isfield(opt,'randomWalkKernelFile'))
        % -- Compute Random Walk kernel
        if(isfield(opt,'graphs'))

    %         K_rw = RWkernel(opt.graphs,10^(-5));
            d = 0;
            for i=1:N
                d = max(d,max(sum(data{i},1)));
            end
            lambda = 10^floor(log10(1/d^2)); % heuristic from RWkernel
            
            if(opt.verbose)
                fprintf(1,'Computing TDP kernel...\n');
            end

            [K_rw, runtime] = tdp_kernel(opt.graphs,lambda,1000);
            runtimes.rw = runtime;
            outname = sprintf('%s/%s_randomwalk.mat',outputFolder,datasetName);
            save(outname,'K_rw','runtime');
            
            Kout.rw = K_rw;
        else
            fprintf(2,'No RW parameters supplied. Cant run RW kernel.\n');
        end    
    else
        load(opt.randomWalkKernelFile);
    end
end

% ------------ LOVASZ THETA KERNEL --------------------

if(useAll || sum(ismember(opt.kernelsUsed,'mec')))
    if(~isfield(opt,'mecKernelFile'))
        % -- Compute Lovasz Labellings
        if(~isfield(opt,'lovaszLabelling'))
            labellingFile = sprintf('%s/%s_lo_labellings.mat',outputFolder,datasetName);
            Ls = lovasz_set(opt.graphs,labellingFile);
            opt.lovaszLabelling = Ls;
        else
            Ls = opt.lovaszLabelling;
        end
        U = {Ls.U};
        K_mec = mec_kernel(U,opt.mecSamples,opt.mecSigma,opt.lovaszParam);
    else
        load(opt.mecKernelFile); 
    end
end

% ------------ SHORTEST-PATH KERNEL --------------------

if(useAll || sum(ismember(opt.kernelsUsed,'sp')))
    if(~isfield(opt,'shortestPathKernelFile'))
        % -- Compute Shortest Path Kernel
    %     if(~isfield(opt,'shortestPaths'))
    %         fprintf(1,'Computing shortest paths...\n')
    %         S = cell(1,N);
    %         maxD = 0;
    %         for i=1:N      
    %             W = data{i};
    %             S{i} = floydWarshall(data{i},W);
    %             Si = S{i};
    %             Siv = Si(:);
    %             maxD = max(max(Siv(~isinf(Siv))),0);
    %         end
    %     else
    %         S = opt.shortestPaths;
    %     end
    %     K_sp = zeros(N,N);
    %     K_sp_n = zeros(N,N); % with normalization
    %     for i=1:N
    %         for j=i:N
    %             Si = S{i}; Sj = S{j};            
    %             hi = histc(Si(:),[1:maxD inf]);
    %             hj = histc(Sj(:),[1:maxD inf]);
    %             k = hi'*hj;
    %             k_n = k/(norm(hi,2)*norm(hj,2));
    %             K_sp(i,j) = k;
    %             K_sp(j,i) = k;
    %             K_sp_n(i,j) = k_n;
    %             K_sp_n(j,i) = k_n;
    %         end
    %     end
        if(opt.verbose)
            fprintf(1,'Computing the Shortest-Path kernel...\n');
        end
        [K_sp runtime sp] = SPkernel(opt.graphs);
        Kout.sp = K_sp;
        stat.sp = struct();
        stat.sp.sp = sp;
        runtimes.sp = runtime;
        outname = sprintf('%s/%s_shortest_path.mat',outputFolder,datasetName);
    %     save(outname,'S','K_sp','K_sp_n');
        save(outname,'K_sp','runtime');
    else
        load(opt.shortestPathKernelFile);
    end
end

% ------------ WEISFEILER-LEHMAN KERNELS --------------------
if(useAll || sum(ismember(opt.kernelsUsed,'wl')))    
    if(~isfield(opt,'wlKernelFile'))
       if(isfield(opt,'graphs'))
          wlopt = process_option_struct(opt.wlopt,...
              {'kernel','iterations'},{'subtree',3});
     
          if(strcmp(wlopt.kernel,'subtree'))
            [K_wls, runtime] = WL(opt.graphs, wlopt.iterations, 1, opt.verbose);
          else
            [K_wls, runtime] = WLspdelta(opt.graphs, wlopt.iterations, 1, 0, opt.verbose);
          end
          K_wl = zeros(N,N);
          for i=1:length(K_wls)
              K_wl = K_wl + K_wls{i};
          end
          
          Kout.wl = K_wl;
          outname = sprintf('%s/%s_weisfeiler-lehman.mat',outputFolder,datasetName);
            save(outname,'K_wl');
          runtimes.wl = runtime;
       else
           fprintf(2,'No WL parameters supplied. Cant run WF kernel.\n');
       end
    else
        load(opt.wlKernelFile);
    end
end

% ------------ LOVASZ SPHERICAL KERNEL ---------------
if(useAll || sum(ismember(opt.kernelsUsed,'spherical')))  
    % -- Compute Lovasz Labellings
    if(~isfield(opt,'lovaszLabelling'))
        labellingFile = sprintf('%s/%s_lo_labellings.mat',outputFolder,datasetName);
        Ls = lovasz_set(opt.graphs,labellingFile);
        opt.lovaszLabelling = Ls;
    else
        Ls = opt.lovaszLabelling;
    end
    Us = {};
    N = length(Ls);
    for i=1:N;
        Us{i} = Ls(i).U;
    end
    
    d = 0;
    for i=1:N;
        d = max(d,size(Us{i},1));
    end
    
    Vs = {};
    for i=1:N
        Xi = Ls(i).X;
        Vs{i} = theta_labelling(Xi,Ls(i).T(i),d,true);
    end
    K_lo_sph = sphericalKernel(Vs);
    outname = sprintf('%s/%s_lovasz_spherical.mat',outputFolder,datasetName);
    save(outname,'K_lo_sph');
end

% ------------ LOVASZ DEGREE KERNEL ---------------
if(useAll || sum(ismember(opt.kernelsUsed,'lovasz-degree')))  
    if(~isfield(opt,'lovaszDegreeKernelFile'))
        [K_lo_deg, Us_eq] = degreeLovaszKernel(data);
        outname = sprintf('%s/%s_graphlet.mat',outputFolder,datasetName);
        save(outname,'K_lo_deg','Us_eq');
    else    
        load(opt.lovaszDegreeKernelFile);
        K_lo_deg = max(K_lo_deg(:))-K_lo_deg;
    end
end

% ------------ LOVASZ FRAIKIN KERNEL --------------------

if(useAll || sum(ismember(opt.kernelsUsed,'lovasz-fraikin'))) 
    if(~isfield(opt,'fraikinKernelFile'))
        % -- Compute Lovasz Labellings
        if(~isfield(opt,'lovaszLabelling'))
            labellingFile = sprintf('%s/%s_lo_labellings.mat',outputFolder,datasetName);
            Ls = lovasz_set(opt.graphs,labellingFile);
            opt.lovaszLabelling = Ls;
        else
            Ls = opt.lovaszLabelling;
        end

        % -- Compute Fraikin kernel
        if(opt.verbose)
            fprintf(1,'Computing Fraikin kernels...\n');
        end
        [K_fr, K_lo, K_lo2] = fraikinKernels(data,{Ls.X},{Ls.U});
        
        outname = sprintf('%s/%s_lovasz_fraikin.mat',outputFolder,datasetName);
        save(outname,'Ls','K_fr','K_lo','K_lo2');
    else    
        % --- Load labelling    
        load(opt.fraikinKernelFile);
    end
end


% ------------ PERFORM CROSS-VALIDATION --------------------
R = {};
if(opt.classify)
    cs = [0.0005 0.001 0.01 0.1 1 2 5];
    gs = [1/N];

    R = struct('cv',[],'c',[],'g',[],'cvs',[],'label',[]);
    R(1) = [];
    
    kexp = struct('name',[],'kernelvar',[],'label',[]);
    kexp(1) = [];
    
    kexp(end+1) = struct('name','sp','kernelvar',...
        'K_sp','label','Shortest Path Kernel');
    
    kexp(end+1) = struct('name','rw','kernelvar',...
        'K_rw','label','Random Walk Kernel');
    
    kexp(end+1) = struct('name','gl','kernelvar',...
        'K_gl','label','Graphlet Kernel');
    
    kexp(end+1) = struct('name','svm','kernelvar',...
        'K_svm','label','SVM-theta Kernel');
    
    kexp(end+1) = struct('name','wl','kernelvar',...
        'K_wl','label','Weisfeiler-Lehman Kernel');
    
    kexp(end+1) = struct('name','mec','kernelvar',...
        'K_mec','label','MEC Kernel');
    
    for iexp=1:length(kexp)
        if(useAll || sum(ismember(opt.kernelsUsed,kexp(iexp).name)))
            Ktrain = [(1:N)' eval(kexp(iexp).kernelvar)];
            r = cross_validate_fit(labels,Ktrain,'-q -t 4 -v 10',cs,gs);
            r.label = ['',kexp(iexp).label];
            R(end+1) = r;
        end
    end

    % ---- VARIATIONS

    if(useAll || sum(ismember(opt.kernelsUsed,'sp_norm')))
        Ktrain_sp_n = [(1:N)' K_sp_n];
        r = cross_validate_fit(labels,Ktrain_sp_n,'-q -t 4 -v 10',cs,gs);
        r.label = 'Shortest Path Kernel (normalized)';
        R(end+1) = r;
    end
    if(useAll || sum(ismember(opt.kernelsUsed,'svm_simple')))
        Ktrain_svm_s = [(1:N)' K_svm_s];
        r = cross_validate_fit(labels,Ktrain_svm_s,'-q -t 4 -v 10',cs,gs);
        r.label = 'SVM-theta Simple Kernel';
        R(end+1) = r;
    end
    if(useAll || sum(ismember(opt.kernelsUsed,'svm_simple_label')))
        Ktrain_svm_sl = [(1:N)' K_svm_sl];
        r = cross_validate_fit(labels,Ktrain_svm_sl,'-q -t 4 -v 10',cs,gs);
        r.label = 'SVM-theta Simple Kernel Label';
        R(end+1) = r;
    end

    % ---- OLD THINGS

    if(useAll || sum(ismember(opt.kernelsUsed,'lovasz')))    
        Ktrain_lo_pm_l = [(1:N)' K_lo_pm_l];
        r = cross_validate_fit(labels,Ktrain_lo_pm_l,'-q -t 4 -v 10',cs,gs);
        r.label = 'Label histogram kernel';
        R(end+1) = r;

        Ktrain_lo_deg = [(1:N)' K_lo_deg];
        r = cross_validate_fit(labels,Ktrain_lo_deg,'-q -t 4 -v 10',cs,gs);
        r.label = 'Lovasz Kernel (Equal-Degree)';
        R(end+1) = r;

        Ktrain_fr = [(1:N)' K_fr];
        r = cross_validate_fit(labels,Ktrain_fr,'-q -t 4 -v 10',cs,gs);
        r.label = 'Fraiken Kernel';
        R(end+1) = r;

        Ktrain_lo = [(1:N)' K_lo];
        r = cross_validate_fit(labels,Ktrain_lo,'-q -t 4 -v 10',cs,gs);
        r.label = 'Lovasz Kernel (X)';
        R(end+1) = r;

        Ktrain_lo2 = [(1:N)' K_lo2];
        r = cross_validate_fit(labels,Ktrain_lo2,'-q -t 4 -v 10',cs,gs);
        r.label = 'Lovasz Kernel (U)';
        R(end+1) = r;

        fprintf(1,'\n');
        fprintf(1,'Lovasz Kernel (Spherical) \n');
        Ktrain_lo_sph = [(1:N)' K_lo_sph];
        [cv_lo_sph c g] = cross_validate_fit(labels,Ktrain_lo_sph,'-q -t 4 -v 10',cs,gs);
        fprintf(1,'Best CV: %.3f%%, c=%.3f, g=%.3f\n',cv_lo_sph,c,g);
    end

    % ---- PYRAMID MATCH

    if(useAll || sum(ismember(opt.kernelsUsed,'pyramidmatch')))
        K_A_pm = importdata('~/programdev/libpmk2/libpmk_mutag/nci1_A/kernel.txt');
        K_A_pmid = importdata('~/programdev/libpmk2/libpmk_mutag/nci1_A/kernel_inputdep.txt');
        K_A_pm_sph = importdata('~/programdev/libpmk2/libpmk_mutag/nci1_A/kernel_sph.txt');

        Ktrain_lo_pm = [(1:N)' K_lo_pm];
        r = cross_validate_fit(labels,Ktrain_lo_pm,'-q -t 4 -v 10',cs,gs);
        r.label = 'Lovasz Kernel (Pyramid-Uniform)';
        R(end+1) = r;

        Ktrain_lo_pmid = [(1:N)' K_lo_pmid];
        r = cross_validate_fit(labels,Ktrain_lo_pmid,'-q -t 4 -v 10',cs,gs);
        r.label = 'Lovasz Kernel (Pyramid-InputDep)';
        R(end+1) = r;

        Ktrain_lo_pm_sph = [(1:N)' K_lo_pm_sph];
        r = cross_validate_fit(labels,Ktrain_lo_pm_sph,'-q -t 4 -v 10',cs,gs);
        r.label = 'Lovasz Kernel (Pyramid-Spherical)';
        R(end+1) = r;

        Ktrain_lo_pm_label = [(1:N)' K_lo_pm_label];
        r = cross_validate_fit(labels,Ktrain_lo_pm_label,'-q -t 4 -v 10',cs,gs);
        r.label = 'Lovasz Kernel Labeled (Pyramid-Uniform)';
        R(end+1) = r;

        Ktrain_lo_pmid_label = [(1:N)' K_lo_pmid_label];
        r = cross_validate_fit(labels,Ktrain_lo_pmid_label,'-q -t 4 -v 10',cs,gs);
        r.label = 'Lovasz Kernel Labeled (Pyramid-InputDep)';
        R(end+1) = r;

        Ktrain_lo_pm_sph_label = [(1:N)' K_lo_pm_sph_label];
        r = cross_validate_fit(labels,Ktrain_lo_pm_sph_label,'-q -t 4 -v 10',cs,gs);
        r.label = 'Lovasz Kernel Labeled (Pyramid-Spherical)';
        R(end+1) = r;

        Ktrain_A_pm = [(1:N)' K_A_pm];
        r = cross_validate_fit(labels,Ktrain_A_pm,'-q -t 4 -v 10',cs,gs);
        r.label = 'Adjacency (Pyramid-Uniform)';
        R(end+1) = r;

        Ktrain_A_pmid = [(1:N)' K_A_pmid];
        r = cross_validate_fit(labels,Ktrain_A_pmid,'-q -t 4 -v 10',cs,gs);
        r.label = 'Adjacency (Pyramid-InputDep)';
        R(end+1) = r;

        Ktrain_A_pm_sph = [(1:N)' K_A_pm_sph];
        r = cross_validate_fit(labels,Ktrain_A_pm_sph,'-q -t 4 -v 10',cs,gs);
        r.label = 'Adjacency (Pyramid-Spherical)';
        R(end+1) = r;

        Ktrain_LS_pm = [(1:N)' K_LS_pm];
        r = cross_validate_fit(labels,Ktrain_LS_pm,'-q -t 4 -v 10',cs,gs);
        r.label = 'LS-Labelling (Pyramid-Uniform)';
        R(end+1) = r;

        Ktrain_LS_pmid = [(1:N)' K_LS_pmid];
        r = cross_validate_fit(labels,Ktrain_LS_pmid,'-q -t 4 -v 10',cs,gs);
        r.label = 'LS-Labelling (Pyramid-InputDep)';
        R(end+1) = r;

        Ktrain_LS_pm_sph = [(1:N)' K_LS_pm_sph];
        r = cross_validate_fit(labels,Ktrain_LS_pm_sph,'-q -t 4 -v 10',cs,gs);
        r.label = 'LS-Labelling (Pyramid-Spherical)';
        R(end+1) = r;
    end
end

if(opt.outputMat)
    outputFileName = sprintf('%s/results_%s.mat',outputFolder,datasetName);
    outputFileName = filename_increment(outputFileName);
    save(outputFileName,'R','opt','runtimes');
end

% logFileName = sprintf('%s/results_%s.txt',outputFolder,datasetName);
% logFileName = filename_increment(logFileName);

% L = log_experiment(opt,R,runtimes);
% f = fopen(logFileName,'w');
% fprintf(f,'%s',L);
% fclose(f);
