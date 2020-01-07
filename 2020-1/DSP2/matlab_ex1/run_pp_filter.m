function y = run_pp_filter(x, pp_filter, L, M)
    %RUN_PP_FILTER Summary of this function goes here
    %   Detailed explanation goes here
    [r,Q] = size(pp_filter);
    len_out = ceil(length(x)*L/M);
    x = padarray(x,[ceil(Q/2), 0]);
    y = zeros(len_out,1);
    for n = 0:len_out-1
        curr_x = floor(n*M/L);
        curr_buffer = x(curr_x+1: curr_x+Q);
        curr_filter = pp_filter(mod(n*M,L)+1,:);
        v = dot(curr_filter, curr_buffer );
        y(n+1,1) = v;
    end
end

