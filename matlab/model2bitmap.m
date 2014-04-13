function [ bitmap ] = model2bitmap( model, n_rows, n_columns )
[height, width, unused] = size(model);
bitmap = zeros(n_rows, n_columns);
for i = 1 : n_rows
    for j =  1 : n_columns
        i_start = round(height / n_rows * (i - 1)) + 1;
        i_end = round(height / n_rows * i);
        j_start = round(width / n_columns * (j - 1)) + 1;
        j_end = round(width / n_columns * j);
        block = model(i_start : i_end, j_start : j_end, :);
        nothing = (block(:,:,1) == 0 ) & (block(:,:,2) == 0 ) & (block(:,:,3) == 0 );
        white = (block(:,:,1) > 180 ) & (block(:,:,2) > 180 ) & (block(:,:,3) > 180 );
        green = (block(:,:,2) - block(:,:,1) > 50 ) & (block(:,:,2) - block(:,:,3) > 50 );
        yellow = (block(:,:,1) - block(:,:,3) > 50 ) & (block(:,:,2) - block(:,:,3) > 50 );
        red = (block(:,:,1) - block(:,:,2) > 50 ) & (block(:,:,1) - block(:,:,3) > 50 );
        blue = (block(:,:,3) - block(:,:,1) > 50 ) & (block(:,:,3) - block(:,:,2) > 50 );
        black = (block(:,:,1) < 80 ) & (block(:,:,2) < 80 ) & (block(:,:,3) < 80 );
        black = black & ~nothing;
        counts = [length(find(nothing)), length(find(white)), length(find(green)), length(find(yellow)), length(find(red)), length(find(blue)), length(find(black))]; % nothing, white, green, yellow, red, blue, black
        n_pixels = sum(counts);
        [color_max, color_idx] = max(counts);
        if color_max / n_pixels > 0.5
            bitmap(i, j) = color_idx;
        else
            bitmap(i, j) = 8; % bad
        end;
    end;
end;
end
