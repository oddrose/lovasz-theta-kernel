function  progresscount(jj,minjj,maxjj,lastjj,txtstr)

if(nargin<5)
    txtstr = 'Progress: ';
end
if(nargin<4)
    lastjj = jj-1;
end
  
if(jj  ~=  minjj)
    lastval = floor(1000*lastjj/maxjj)/10;
    curval = floor(1000*jj/maxjj)/10;
    if(lastval~=curval)
    %     laststr = sprintf('%d/%d',lastjj,maxjj); % ints
        laststr = sprintf('%.1f%%',lastval);
        fprintf(1,repmat('\b',1,length(laststr)));
    %     fprintf(1,'%d/%d',jj,maxjj); % ints
        fprintf(1,'%.1f%%',curval);
    end
else
%     fprintf(1,'%s%d/%d',txtstr,minjj,maxjj); % ints
    fprintf(1,'%s%.1f%%',txtstr,floor(1000*jj/maxjj)/10);
end
if(jj  ==  maxjj)
    fprintf(1,'\n');
end
  