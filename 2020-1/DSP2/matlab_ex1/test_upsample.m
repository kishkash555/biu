
L=7; 
M=5;
% x = audioread("doors.wav");
% x = x(2001:3000);
% x = x';


h = create_lpf(L,1);
xi = zeros(length(x)*L,1);
xi(1:L:end) = x;

nv = conv(h,xi);
y = pp_interpolate(x,h,L);


%>> norm(y(1:7200)-nv)
% 
% ans =
% 
%    1.1927e-15
% 

