function v1 = padvec(v,c)
v1 = zeros(length(v)+c,1);
v1(1:length(v))=v;
