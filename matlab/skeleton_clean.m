function [ bw_skel ] = skeleton_clean( bw )
bw_skel = bwmorph(bw, 'skel', Inf);

% cleaning skeleton...
for iter = 1 : 5
    branchpoints = bwmorph(bw_skel, 'branchpoints');
    endpoints = bwmorph(bw_skel, 'endpoints');
    endpoints_idxes = find(endpoints);
    branchpoints_dilated = imdilate(branchpoints, strel([0 1 0; 1 1 1; 0 1 0]));
    bw_skel_splitted = bw_skel;
    bw_skel_splitted(branchpoints_dilated) = 0;
    CC = bwconncomp(bw_skel_splitted);
    num_pixels = cellfun(@numel,CC.PixelIdxList);
    for i = 1 : length(num_pixels)
        if num_pixels(i) > 10
            continue;
        end;
        l = CC.PixelIdxList{i};
        for j = 1 : length(l);
            if any(endpoints_idxes == l(j))
                bw_skel(l) = 0;
                break;
            end;
        end;
    end;
    bw_skel = bwmorph(bw_skel, 'shrink', 1);
    bw_skel = bwmorph(bw_skel, 'skel', Inf);
end;

end

