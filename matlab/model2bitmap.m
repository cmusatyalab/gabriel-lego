function [ bitmap, row_compensates ] = model2bitmap( model, n_rows, n_columns, row_compensates )
[height, width, unused] = size(model);
bitmap = zeros(n_rows, n_columns);
for i = 1 : n_rows
    for j =  1 : n_columns
        i_start = round(height / n_rows * (i - 1)) + 1;
        i_end = round(height / n_rows * i);
        j_start = round((width - sum(row_compensates(i, :))) / n_columns * (j - 1)) + 1 + row_compensates(i, 1);
        j_end = round((width - sum(row_compensates(i, :))) / n_columns * j) + row_compensates(i, 1);
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

row_compensates = ones(n_rows, 2) * (-1);
for i = 1 : n_rows
    i_start = round(height / n_rows * (i - 1)) + 1;
    i_end = round(height / n_rows * i);

    i1 = round(i_start / 3 + i_end / 3 * 2);
    leading1 = 1;
    while sum(model(i1, leading1, :)) == 0
        leading1 = leading1 + 1;
    end;
    trailing1 = 1;
    while sum(model(i1, width - trailing1 + 1, :)) == 0
        trailing1 = trailing1 + 1;
    end;
    leading1 = leading1 - 1;
    trailing1 = trailing1 - 1;

    i2 = round(i_start / 3 * 2 + i_end / 3);
    leading2 = 1;
    while sum(model(i2, leading2, :)) == 0
        leading2 = leading2 + 1;
    end;
    trailing2 = 1;
    while sum(model(i2, width - trailing2 + 1, :)) == 0
        trailing2 = trailing2 + 1;
    end;
    leading2 = leading2 - 1;
    trailing2 = trailing2 - 1;

    i3 = round((i_start + i_end) / 2);
    leading3 = 1;
    while sum(model(i3, leading3, :)) == 0
        leading3 = leading3 + 1;
    end;
    trailing3 = 1;
    while sum(model(i3, width - trailing3 + 1, :)) == 0
        trailing3 = trailing3 + 1;
    end;
    leading3 = leading3 - 1;
    trailing3 = trailing3 - 1;

    leading = (leading1 + leading2 + leading3) / 3;
    trailing = (trailing1 + trailing2 + trailing3) / 3;
    row_compensates(i, :) = [leading, trailing];
    
    if bitmap(i, 1) == 1
        row_compensates(i, 1) = -1;
    end;
    if bitmap(i, n_columns) == 1
        row_compensates(i, 2) = -1;
    end;
end;
for i = 1 : n_rows
    if row_compensates(i, 1) == -1
        j1 = i - 1;
        while j1 >= 1
            if row_compensates(j1, 1) ~= -1
                break;
            end;
            j1 = j1 - 1;
        end;
        j2 = i + 1;
        while j2 <= n_rows
            if row_compensates(j2, 1) ~= -1
                break;
            end;
            j2 = j2 + 1;
        end;
        if j1 < 1 || (j2 <= n_rows && (j2 - i) <= (i - j1))
            row_compensates(i, 1) = row_compensates(j2, 1);
        else
            row_compensates(i, 1) = row_compensates(j1, 1);
        end;
    end;
    if row_compensates(i, 2) == -1
        j1 = i - 1;
        while j1 >= 1
            if row_compensates(j1, 2) ~= -1
                break;
            end;
            j1 = j1 - 1;
        end;
        j2 = i + 1;
        while j2 <= n_rows
            if row_compensates(j2, 2) ~= -1
                break;
            end;
            j2 = j2 + 1;
        end;
        if j1 < 1 || (j2 <= n_rows && (j2 - i) <= (i - j1))
            row_compensates(i, 2) = row_compensates(j2, 2);
        else
            row_compensates(i, 2) = row_compensates(j1, 2);
        end;
    end;
end;

end
