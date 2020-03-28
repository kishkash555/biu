function y=pp_decimate(x,h,M)
h = pad_divisible(h,M);
x = pad_divisible(x,M);

h_pp = reshape(h, M, length(h)/M);
x_pp = reshape(x, M, length(x)/M);

y = zeros(1,size(h_pp,2)+size(x_pp,2)-1);

for i=1:M
    new_comp = conv(h_pp(i,:),x_pp(i,:));
    new_comp = circshift(new_comp,i);
    y = y + new_comp;
end
    y=circshift(y,-1);
end