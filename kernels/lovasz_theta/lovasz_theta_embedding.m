function [Ls, runtime] = lovasz_theta_embedding(graphs,outfile,resume)

cvx_clear

N = length(graphs);

Us = {};
cs = {};
Xs = {};
ts = {};

if(nargin<3)
    resume = false;
end

if(resume)
    if(exist(outfile,'file'))
        load(outfile);
    end
end

if(exist('last','var'))
    last0 = last;
else
    last0 = length(Us);
end

runtimes = {};
runtime = 0;

fprintf(1,'Computing Lovasz labellings...\n');
for i=(last0+1):N
    fprintf(1,'%d/%d\n',i,N);
    A = graphs(i).am;
    t0 = cputime;
    [Us{i}, cs{i}, ts{i}, Xs{i}] = lovasz_theta_vectors(A);
    runtimes{i} = cputime-t0;
    last = i;
    save(outfile,'Us','cs','ts','Xs','runtimes','last');
    runtime = runtime + runtimes{i};
end
Ls = struct('U',Us,'c',cs,'t',ts,'X',Xs,'runtime',runtimes);

save(outfile,'Ls','cs','ts','Xs','last');
