clc;

image_path = '/home/zhuoc/Workspace/gabriel/src/app/lego/test_images/frame-030.jpeg';

im_rgb = imread(image_path);
im_bw = rgb2gray(im_rgb);
[row, column] = size(im_bw);

figure;
subplot(2, 2, 1); subimage(im_rgb);

%% detecting a specific color (blue here)
blue_mask = (im_rgb(:,:,3) - im_rgb(:,:,1) >= 20 ) & (im_rgb(:,:,3) - im_rgb(:,:,2) >= 20 );
im_rgb_blue = im_rgb;
im_rgb_blue(repmat(~blue_mask,[1 1 3])) = 0;
subplot(2, 2, 2); subimage(im_rgb_blue);

%% edge detection
[im_edge, thresh] = edge(im_bw);
subplot(2, 2, 3); subimage(im_edge);

%% detecting black dots
black_mask = (im_rgb(:,:,1) <= 100 ) & (im_rgb(:,:,2) <= 100 ) & (im_rgb(:,:,3) <= 100 ) & (abs(im_rgb(:,:,1) - im_rgb(:,:,2)) <= 20) & (abs(im_rgb(:,:,2) - im_rgb(:,:,3)) <= 20 ) & (abs(im_rgb(:,:,1) - im_rgb(:,:,3)) <= 20 );
im_rgb_black = im_rgb;
im_rgb_black(repmat(~black_mask,[1 1 3])) = 0;
%figure; imshow(im_rgb_black);
CC = bwconncomp(black_mask);
num_pixels = cellfun(@numel,CC.PixelIdxList);
mask_black_dots = black_mask;
for i = 1 : length(num_pixels)
    if num_pixels(i) > 20
        l = CC.PixelIdxList{i};
        mask_black_dots(l) = 0;
    else
        l = CC.PixelIdxList{i};
        [rows, columns] = ind2sub(CC.ImageSize, l);
        if max(rows) - min(rows) > 5 || max(columns) - min(columns) > 5
            mask_black_dots(l) = 0;
        end;
    end;
end;
im_rgb_black_dots = im_rgb;
im_rgb_black_dots(repmat(~mask_black_dots,[1 1 3])) = 0;
%figure; imshow(im_rgb_black_dots);

%% detect board
% count black dots in each block...
counts = zeros(9, 16);
CC = bwconncomp(mask_black_dots);
for i = 1 : CC.NumObjects
    l = CC.PixelIdxList{i};
    [rows, columns] = ind2sub(CC.ImageSize, l);
    mean_row = mean(rows);
    mean_column = mean(columns);
    counts(ceil(mean_row / 80), ceil(mean_column / 80)) = counts(ceil(mean_row / 80), ceil(mean_column / 80)) + 1;
end;
[max_i, idx_i] = max(counts);
[max_value, idx_j] = max(max_i);
idx_i = idx_i(idx_j);
% this should be a point in the board
board_point = [80 * (idx_i - 1) + 40, 80 * (idx_j - 1) + 40];

% now detect board border
CC = bwconncomp(black_mask);
num_pixels = cellfun(@numel,CC.PixelIdxList);
min_dist = 10000;
min_idx = 0;
for i = 1 : length(num_pixels)
    if num_pixels(i) > 100
        l = CC.PixelIdxList{i};
        mask_board_border_tmp = logical(zeros(row, column));
        mask_board_border_tmp(l) = 1;
        mask_skel = bwmorph(mask_board_border_tmp, 'skel', Inf);
        [rows, columns] = find(mask_skel);
        if max(rows) - min(rows) > 100 && max(columns) - min(columns) > 100
            mean_row = mean(rows);
            mean_column = mean(columns);
            d = pdist2([mean_row, mean_column], board_point);
            if d < min_dist
                min_dist = d;
                min_idx = i;
            end;
        end;
    end;
end;
mask_board_border = logical(zeros(row, column));
mask_board_border(CC.PixelIdxList{min_idx}) = 1;

% detect board
mask_board_border_skel = skeleton_clean(mask_board_border);
endpoints = bwmorph(mask_board_border_skel, 'endpoints');
[endpoints_I, endpoints_J] = find(endpoints);
if ~isempty(endpoints_I)
    endpoints1 = [endpoints_I(1), endpoints_J(1)];
    endpoints2 = [endpoints_I(2), endpoints_J(2)];
    y = linspace(endpoints1(1), endpoints2(1), 200);
    x = (endpoints1(2) - endpoints2(2)) / (endpoints1(1) - endpoints2(1)) * (y - endpoints1(1)) + endpoints1(2);
    index = sub2ind([row, column], round(y), round(x));
    mask_board_border_skel(index) = 1;
end;
mask_board = imfill(mask_board_border_skel, 'holes');
    
im_rgb_board = im_rgb;
im_rgb_board(repmat(~mask_board,[1 1 3])) = 0;
im_bw_board = im_bw;
im_bw_board(~mask_board) = 0;
[rows, columns] = find(mask_board);
board_center = [(min(rows) + max(rows)) / 2, (min(columns) + max(columns)) / 2];

%% detect model
edge_board = edge(im_bw_board);
edge_board = imdilate(edge_board, strel(ones(6)));
edge_board = imerode(edge_board, strel(ones(6)));
edge_board = bwmorph(edge_board, 'open');

mask_model = ~edge_board & mask_board;

CC = bwconncomp(mask_model);
num_pixels = cellfun(@numel,CC.PixelIdxList);
min_dist = 10000;
min_idx = 0;
for i = 1 : length(num_pixels)
    if num_pixels(i) < 200
        continue;
    end;
    l = CC.PixelIdxList{i};
    [rows, columns] = ind2sub([row, column], l);
    mean_row = mean(rows);
    mean_column = mean(columns);
    d = pdist2([mean_row, mean_column], board_center);
    if d < min_dist
        min_dist = d;
        min_idx = i;
    end;
end;
mask_model = logical(zeros(row, column));
mask_model(CC.PixelIdxList{min_idx}) = 1;
for iter = 1 : 5
    mask_model = imerode(mask_model, [0 0 0;0 1 0;0 1 0]);
end;
im_rgb_model = im_rgb;
im_rgb_model(repmat(~mask_model,[1 1 3])) = 0;

%% adjust orientation
[Gmag, Gdir] = imgradient(mask_model);
dirs = Gdir(find(Gmag));
dirs(dirs < 0) = dirs(dirs < 0) + 180;
dirs = dirs - 90;
dirs(dirs < 0) = dirs(dirs < 0) + 90;
mean1 = mean(dirs);
std1 = std(dirs);
dirs2 = dirs - 45;
dirs2(dirs2 < 0) = dirs2(dirs2 < 0) + 90;
mean2 = mean(dirs2);
std2 = std(dirs2);
if std1 < std2
    dirs(dirs > (mean1 + 45)) = dirs(dirs > (mean1 + 45)) - 90;
    dirs(dirs < (mean1 - 45)) = dirs(dirs < (mean1 - 45)) + 90;
    mean1 = mean(dirs(dirs < (mean1 + 35) & dirs > (mean1 - 35)));
    result = mean1;
else
    dirs2(dirs2 > (mean2 + 45)) = dirs2(dirs2 > (mean2 + 45)) - 90;
    dirs2(dirs2 < (mean2 - 45)) = dirs2(dirs2 < (mean2 - 45)) + 90;
    mean2 = mean(dirs2(dirs2 < (mean2 + 35) & dirs2 > (mean2 - 35)));
    result = mean2 + 45;
end;
if result > 90
    result = result - 90;
end;
if result < 0
    result = result + 90;
end;
if result > 45
    result = restul - 90;
end;
im_rgb_model = imrotate(im_rgb_model, -result);
mask_model = imrotate(mask_model, -result);
[rows, columns] = find(mask_model);

%% crop model
model_cropped = im_rgb_model(min(rows) : max(rows), min(columns) : max(columns), :);
model_mask_cropped = mask_model(min(rows) : max(rows), min(columns) : max(columns), :);

%% now convert model to discreet map representation
min_bad = 1000;
min_bad_n_rows = 0;
min_bad_n_columns = 0;
min_bad_bitmap = NaN;
for n_rows = 10 : 10
    for n_columns = 6 : 6
        row_compensates = zeros(n_rows, 2);
        [bitmap, row_compensates] = model2bitmap(model_cropped, n_rows, n_columns, row_compensates);
        [bitmap, row_compensates] = model2bitmap(model_cropped, n_rows, n_columns, row_compensates);
        n_bad = length(find(bitmap == 8));
        if n_bad / n_rows / n_columns < min_bad
            min_bad = n_bad / n_rows / n_columns;
            min_bad_n_rows = n_rows;
            min_bad_n_columns = n_columns;
            min_bad_bitmap = bitmap;
        end;
    end;
end;
synthetic_model = paint_bitmap(min_bad_bitmap);