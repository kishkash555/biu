
L=2; 
M=5;
fs=8000;
x = audioread("doors.wav");
%x = x(2001:3000);
x = x';

y = pp_resample(x,L,M);

soundsc(y,fs*L/M);
h = create_lpf(L,M);
%nv = upfirdn(x,h,L,M);
nv = resample(x,L,M);
minlen=min(length(nv), length(y));
df=nv(1,1:minlen)-y(1,1:minlen);
norm(df)/norm(nv)
% h = create_lpf(L,1);
% xi = zeros(length(x)*L,1);
% xi(1:L:end) = x;
% 
% nv = conv(h,xi);
% y = pp_interpolate(x,h,L);
% 
