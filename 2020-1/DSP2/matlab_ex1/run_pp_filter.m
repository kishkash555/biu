function y = run_pp_filter(x, pp_filter, L, M)
    n_filters = size(pp_filter,1);
    siglen = size(x,2);
    if mod(siglen,n_filters) ~= 0
        padlen = siglen-mod(siglen,n_filters);
        x = padarray(x, [0, padlen],'post');
        siglen = size(x,2);
    end
    indL = mod(0:(M*L-1),L);
    indM = mod(0:(M*L-1),M);
    
end