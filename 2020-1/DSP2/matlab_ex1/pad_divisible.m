function y = pad_divisible(v, M)
% pad zeros at the end of v to make its length divisible by M
s=size(v);
padlen = mod(M- mod(s(2),M),M);
if padlen > 0
    y = zeros(s(1),s(2)+padlen);
    y(:,1:s(2))=v;
else
    y=v;
end

    
    