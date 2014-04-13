function [ model ] = paint_bitmap( bitmap )
block_rows = 25;
block_columns = 20;
model = zeros(size(bitmap, 1) * block_rows, size(bitmap, 2) * block_columns, 3);
for i = 1 : size(bitmap, 1)
    for j = 1 : size(bitmap, 2)
        % nothing, white, green, yellow, red, blue, black
        block = zeros(block_rows, block_columns, 3);
        if bitmap(i, j) == 2
            block(:, :, :) = 255;
        elseif bitmap(i, j) == 3
            block(:, :, 2) = 255;
        elseif bitmap(i, j) == 4
            block(:, :, 1 : 2) = 255;
        elseif bitmap(i, j) == 5
            block(:, :, 1) = 255;
        elseif bitmap(i, j) == 6
            block(:, :, 3) = 255;
        elseif bitmap(i, j) == 7
            block(:, :, :) = 0;
        else
            block(:, :, :) = 128;
        end;
        model((i - 1) * block_rows + 1 : i * block_rows, (j - 1) * block_columns + 1 : j * block_columns, :) = block;
    end;
end;
model = model / 255;
end

