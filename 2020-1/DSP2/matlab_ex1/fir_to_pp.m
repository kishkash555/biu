function pp_filter = fir_to_pp(L,M, orig_filter)
    %FIR_TO_PP get the filter coefficients in polyphase format
    %   Detailed explanation goes here
    Lh = length(orig_filter);
    if mod(Lh,L) ~= 0
        padlen = L-mod(Lh,L);
        orig_filter = padarray(orig_filter, [0, padlen],'post');
        Lh = length(orig_filter);
    end
   
    pp_filter = reshape(orig_filter,[L ,Lh/L]);
    
    %mM_mod_L = mod((0 : Lh/L)*M,L);
    %pp_filter = g(mM_mod_L+1, :);
end

