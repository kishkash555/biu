function [ws] = synthesis_win(w, L)
N = length(w);
ws = w;
w2 = zeros(N, 1);
for n=0:N-1
    qn = floor(n/L);
    qm = floor((N-1-n)/L);
    for q=-qn:qm
        w2(n+1) = w2(n+1)+w(q*L+n+1)^2;
    end 
end
ws = flipdim(w./w2, 1);
