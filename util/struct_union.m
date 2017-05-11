function S = structunion(S1,S2)

S = S1;

F = fieldnames(S2);

for i=1:length(fieldnames(S2));
   f = F{i};
   S.(f) = S2.(f);
end