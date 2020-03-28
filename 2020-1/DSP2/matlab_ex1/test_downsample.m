
L=7; 
M=6;
% x = audioread("doors.wav");
% x = x(2001:3000);
% x = x';
% h = create_lpf(M,1);

%%% simpler signals
x = [1:100,100:-1:1];
x = [x,x,x,x];
h = 10:-1:1;



h = pad_divisible(h,M);
x = pad_divisible(x,M);

nv = conv(x,h);
nv = nv(1:M:end);
y = pp_decimate2(x,h,M);

minlen = min(length(y),length(nv));
y = y(1,1:minlen);
nv = nv(1,1:minlen);
%sprintf("y %d nv %d",length(y),length(nv))
norm(nv-y)
% norm(nv(2:M:end-3)-y(1:end))
% norm(nv(3:M:end-3)-y)


