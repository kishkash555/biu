
M=2;
x = audioread("doors.wav");
x = x(2001:3000);
x = x';
h = create_lpf(M,1);

%% simpler signals
x = [1:100,100:-1:1];
x = [x,x,x,x];
h = 10:-1:1;

%% Test of pp_decimate
% h = pad_divisible(h,M);
% x = pad_divisible(x,M);
% 
% nv = conv(x,h);
% nv = nv(1:M:end);
% y = pp_decimate(x,h,M);
% 
%% test of brute-force convo
lh = length(h);
x2 = [zeros(1,lh-1), x,zeros(1,lh)];
y = zeros(size(x2)-lh-1);
% full convolution
for shift=1:length(x2)-lh
    y(1,shift)=dot(x2(1,shift:shift+lh-1),h);
end
yc = conv(x,h);
yc = yc(1:2:end);
norm(y-yc)

%% convolution with decimation by 2
ly = floor((length(x2)-lh)/2);
y = zeros(1,ly);

for shift=1:ly
    y(1,shift)=dot(x2(1,shift*2-1:shift*2-1+lh-1),h);
end

yc = conv(x,h);
yc = yc(1:2:end);
norm(y-yc)

%% convolution with decim by 2 
h0 = h(1:2:end);
h1 = h(2:2:end);

h1 = [h1,0];

yt0 = conv(x(1:2:end),h0);
yt1 = conv(x(2:2:end),h1);

yt1 = [0, yt1(1:end-1)];

yc = conv(x,h);
pp= yt0+yt1;
nv = yc(1,1:2:end);
%pp = circshift(pp,1);
norm(pp-nv)


%% polyphase check
lh = length(h);
ht = zeros(1,lh);

ht(1:2:end) = h(1:2:end);
s0 = conv(x,ht);
y0 = s0(1:2:end);


h0 = h(1:2:end); % no spacing
yt = conv(x(1:2:end),h0);

norm(yt-y0) % identical

%% polyphase check
lh = length(h);
ht = zeros(1,lh-1);

ht(1:2:end) = h(2:2:end);
s0 = conv(x,ht);
y1 = [s0(2:2:end),0];


h1 = [h(2:2:end) 0]; % no spacing
yt = conv(x(2:2:end),h1);

norm(yt-y1) 

%% combine checks
h0 = h(1:2:end); % no spacing
h1 = [h(2:2:end) 0]; % no spacing
y0 = conv(x(1:2:end),h0);
y1 = conv(x(2:2:end),h1);

s=conv(x,h);

%% polyphase before
lh = length(h);
ht = zeros(1,lh);
h0 = zeros(1,lh);
ht(1:2:end) = h(1:2:end);
h0(2:2:end) = h(2:2:end);


s0 = conv(x,ht);
s1 = conv(x,h0);
y = s0+s1(1:end);
yc = conv(x,h);

norm(y-yc) % identical

