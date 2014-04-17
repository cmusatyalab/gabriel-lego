function [ tform ] = get_transform_matrix( mask_board )
% input_points = [282, 462; 644, 410; 269, 671; 707, 591]; % im30
input_points = [271, 491; 660, 447; 232, 734; 719, 651]; % im42
base_points = [0, 0; 270, 0; 0, 155; 270, 155];

tform = cp2tform(input_points, base_points, 'projective');

end

