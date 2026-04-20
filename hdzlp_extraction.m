% =========================================================================
% Hist-LDZP Feature Extraction — Tunable Parameters
% KinFaceW-II — Multiple configurations
% =========================================================================
% Extracts Hist-LDZP features using configurable patch_size, step_size,
% top_k (number of dominant Kirsch directions to encode).
% Processes 3 color channels: R, G, B.
% =========================================================================

clc; clear; close all;

% ??? Paths ???????????????????????????????????????????????????????????????????
dataset_path = 'C:\Users\surface laptop 5\Downloads\KinFaceW-II\KinFaceW-II\images';
output_base  = 'C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LDZP';

relations = {'FD', 'FS', 'MD', 'MS'};

% ??? Configuration grid ??????????????????????????????????????????????????????
configs = {
    struct('patch_size', 32, 'step_size', 2, 'top_k', 3, 'tag', 'ps32_ss2_k3');
    struct('patch_size', 16, 'step_size', 2, 'top_k', 3, 'tag', 'ps16_ss2_k3');
    struct('patch_size', 16, 'step_size', 2, 'top_k', 4, 'tag', 'ps16_ss2_k4');
    struct('patch_size', 16, 'step_size', 2, 'top_k', 5, 'tag', 'ps16_ss2_k5');
    struct('patch_size', 32, 'step_size', 2, 'top_k', 4, 'tag', 'ps32_ss2_k4');
};

% ??? Kirsch masks (8 compass directions) ?????????????????????????????????????
kirsch_masks = zeros(3, 3, 8);
kirsch_masks(:,:,1) = [ 5  5  5; -3  0 -3; -3 -3 -3];  % N
kirsch_masks(:,:,2) = [ 5  5 -3;  5  0 -3; -3 -3 -3];  % NE
kirsch_masks(:,:,3) = [ 5 -3 -3;  5  0 -3;  5 -3 -3];  % E
kirsch_masks(:,:,4) = [-3 -3 -3;  5  0 -3;  5  5 -3];  % SE
kirsch_masks(:,:,5) = [-3 -3 -3; -3  0 -3;  5  5  5];  % S
kirsch_masks(:,:,6) = [-3 -3 -3; -3  0  5; -3  5  5];  % SW
kirsch_masks(:,:,7) = [-3 -3  5; -3  0  5; -3 -3  5];  % W
kirsch_masks(:,:,8) = [-3  5  5; -3  0  5; -3 -3 -3];  % NW

% ??? Main loop over configs ???????????????????????????????????????????????????
for ci = 1:length(configs)
    cfg        = configs{ci};
    patch_size = cfg.patch_size;
    step_size  = cfg.step_size;
    top_k      = cfg.top_k;
    tag        = cfg.tag;

    output_path = fullfile(output_base, ['Color_HLDZP_' tag]);
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end

    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('Config: patch_size=%d, step_size=%d, top_k=%d\n', ...
            patch_size, step_size, top_k);
    fprintf('%s\n', repmat('=', 1, 60));

    for r = 1:length(relations)
        rel       = relations{r};
        rel_path  = fullfile(dataset_path, rel);

        files = [dir(fullfile(rel_path, '*.jpg')); ...
                 dir(fullfile(rel_path, '*.png')); ...
                 dir(fullfile(rel_path, '*.jpeg'))];
        files = sort_files(files);

        fprintf('\n  %s — %d images\n', rel, length(files));

        vectors = [];

        for k = 1:length(files)
            img_path = fullfile(rel_path, files(k).name);
            img      = imread(img_path);

            if size(img, 3) == 1
                img = cat(3, img, img, img);
            end

            feat = extract_color_histldzp(img, patch_size, step_size, ...
                                           top_k, kirsch_masks);

            if ~isempty(feat)
                vectors = [vectors; feat];
            end

            if mod(k, 50) == 0 || k == length(files)
                fprintf('    %d/%d processed\n', k, length(files));
            end
        end

        save_path = fullfile(output_path, ['HistLDZP_' rel '.mat']);
        save(save_path, 'vectors');

        fprintf('  Saved ? [%d x %d]\n', size(vectors, 1), size(vectors, 2));
        fprintf('  Range: [%.4f, %.4f]\n', min(vectors(:)), max(vectors(:)));
    end
end

fprintf('\nAll configs extracted.\n');


% =========================================================================
% FUNCTIONS
% =========================================================================

% ??? Sort files alphabetically ???????????????????????????????????????????????
function sorted = sort_files(files)
    names = {files.name};
    [~, idx] = sort(names);
    sorted = files(idx);
end

% ??? Color Hist-LDZP extraction ??????????????????????????????????????????????
function feature = extract_color_histldzp(img, patch_size, step_size, ...
                                            top_k, kirsch_masks)
    R = double(img(:,:,1));
    G = double(img(:,:,2));
    B = double(img(:,:,3));

    feat_R = histldzp_channel(R, patch_size, step_size, top_k, kirsch_masks);
    feat_G = histldzp_channel(G, patch_size, step_size, top_k, kirsch_masks);
    feat_B = histldzp_channel(B, patch_size, step_size, top_k, kirsch_masks);

    if isempty(feat_R) || isempty(feat_G) || isempty(feat_B)
        feature = [];
        return;
    end

    feature = [feat_R, feat_G, feat_B];
end

% ??? Hist-LDZP for single channel ????????????????????????????????????????????
function hist_vec = histldzp_channel(channel, patch_size, step_size, ...
                                      top_k, kirsch_masks)
    [h, w] = size(channel);

    % Compute LDZP map for entire channel at once
    ldzp_map = compute_ldzp_map(channel, top_k, kirsch_masks);

    hist_list = [];

    for i = 1 : step_size : h - patch_size + 1
        for j = 1 : step_size : w - patch_size + 1
            patch = ldzp_map(i:i+patch_size-1, j:j+patch_size-1);
            hist_p = histcounts(patch(:), 256, 'BinLimits', [0, 255]);
            hist_p = double(hist_p);
            s = sum(hist_p);
            if s > 0
                hist_p = hist_p / s;
            end
            hist_list = [hist_list, hist_p];
        end
    end

    hist_vec = hist_list;
end

% ??? Compute LDZP map for entire channel ?????????????????????????????????????
function ldzp_map = compute_ldzp_map(channel, top_k, kirsch_masks)
    [h, w] = size(channel);

    % Compute 8 directional responses
    responses = zeros(h, w, 8);
    for d = 1:8
        responses(:,:,d) = abs(imfilter(channel, kirsch_masks(:,:,d), ...
                                         'same', 'replicate'));
    end

    % Rank directions per pixel — descending
    [~, ranked] = sort(responses, 3, 'descend');  % ranked(i,j,k) = k-th strongest direction

    % Encode top-k directions as binary code
    ldzp_map = zeros(h, w, 'uint8');
    for k = 1:top_k
        dir_map = ranked(:,:,k);  % (h, w) — direction index (1-8)
        for d = 1:8
            ldzp_map(dir_map == d) = ldzp_map(dir_map == d) + uint8(2^(d-1));
        end
    end
end