function [ new_bw ] = fill_in_holes( original, size )
filled = imfill(original, 'holes');
holes = filled & ~original;
bigholes = bwareaopen(holes, size);
smallholes = holes & ~bigholes;
new_bw = original | smallholes;
end

