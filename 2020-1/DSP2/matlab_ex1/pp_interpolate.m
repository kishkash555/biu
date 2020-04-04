function y=pp_interpolate(x,h,L)
h = pad_divisible(h,L);
x = pad_divisible(x,L);

h_pp_size = [L, length(h)/L];
%x_pp_size = [L, length(x)/L];

h_pp = reshape(h, h_pp_size);

y_pp = zeros(L,h_pp_size(2)+length(x)-1);

for i=1:L
    y_pp(i,:) = conv(h_pp(i,:),x); % more appropriate than filter
end
%y=y_pp(:);
y=y_pp;
end