% =========================================================================
% Hist-LDZP Kinship Pipeline — Grid Search over Configs
% KinFaceW-II — 5-fold cross-validation
% =========================================================================

clc; clear; close all;

% ??? Paths ???????????????????????????????????????????????????????????????????
HLDZP_BASE = 'C:\Users\surface laptop 5\OneDrive\Documents\PFE\Methodes classiques\Hist-LDZP';
MAT_DIR    = 'C:\Users\surface laptop 5\OneDrive\Documents\PFE\lbp';

relations  = {'Father-Daughter', 'Father-Son', 'Mother-Daughter', 'Mother-Son'};
rel_codes  = {'FD', 'FS', 'MD', 'MS'};
mat_files  = {'LBP_fd.mat', 'LBP_fs.mat', 'LBP_md.mat', 'LBP_ms.mat'};

CONFIGS    = {'ps32_ss2_k3', 'ps16_ss2_k3', 'ps16_ss2_k4', 'ps16_ss2_k5', 'ps32_ss2_k4'};

C_GRID     = [0.01, 0.1, 1, 10, 100, 1000];

all_results = struct();

for ci = 1:length(CONFIGS)
    config_tag = CONFIGS{ci};
    config_dir = fullfile(HLDZP_BASE, ['Color_HLDZP_' config_tag]);

    if ~exist(config_dir, 'dir')
        fprintf('Skipping %s — directory not found\n', config_tag);
        continue;
    end

    fprintf('\n%s\n', repmat('=', 1, 55));
    fprintf('  Config: %s\n', config_tag);
    fprintf('%s\n', repmat('=', 1, 55));

    config_means = zeros(1, 4);

    for r = 1:4
        rel      = rel_codes{r};
        mat_path = fullfile(MAT_DIR, mat_files{r});
        pkl_path = fullfile(config_dir, ['HistLDZP_' rel '.mat']);

        if ~exist(pkl_path, 'file')
            fprintf('  Skipping %s — file not found\n', rel);
            continue;
        end

        feat_data = load(pkl_path);
        feats     = double(feat_data.vectors);

        meta      = load(mat_path);
        idxa      = double(meta.idxa(:));
        idxb      = double(meta.idxb(:));
        fold      = double(meta.fold(:));
        y         = double(meta.matches(:));

        X = abs(feats(idxa, :) - feats(idxb, :));

        fold_scores = zeros(1, 5);

        for f = 1:5
            train_mask = fold ~= f;
            test_mask  = fold == f;

            X_tr_raw = X(train_mask, :);
            X_te_raw = X(test_mask,  :);
            y_train  = y(train_mask);
            y_test   = y(test_mask);
            fold_ids = fold(train_mask);

            best_C = find_best_C(X_tr_raw, y_train, fold_ids, C_GRID);

            [X_tr, X_te] = power_normalization(X_tr_raw, X_te_raw);

            mdl   = fitcsvm(X_tr, y_train, 'KernelFunction', 'linear', ...
                            'BoxConstraint', best_C, 'Standardize', false, ...
                            'ClassNames', [0, 1]);
            preds = predict(mdl, X_te);
            fold_scores(f) = mean(preds == y_test);
        end

        mean_acc = mean(fold_scores);
        config_means(r) = mean_acc;
        fprintf('  %s: %.2f%%\n', relations{r}, mean_acc * 100);
    end

    overall = mean(config_means);
    fprintf('  Overall: %.2f%%\n', overall * 100);
    all_results.(strrep(config_tag, '-', '_')) = struct(...
        'per_relation', config_means, 'overall', overall);
end

% ??? Summary ??????????????????????????????????????????????????????????????????
fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('  SUMMARY — All Hist-LDZP Configs\n');
fprintf('%s\n', repmat('=', 1, 70));
fprintf('%-20s %8s %8s %8s %8s %8s\n', 'Config', 'FD', 'FS', 'MD', 'MS', 'Global');
fprintf('%s\n', repmat('-', 1, 65));

fields     = fieldnames(all_results);
best_tag   = '';
best_score = -1;

for i = 1:length(fields)
    tag = fields{i};
    res = all_results.(tag);
    pr  = res.per_relation;
    fprintf('%-20s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %7.2f%%\n', ...
            tag, pr(1)*100, pr(2)*100, pr(3)*100, pr(4)*100, res.overall*100);
    if res.overall > best_score
        best_score = res.overall;
        best_tag   = tag;
    end
end

fprintf('\n  Best config : %s ? %.2f%%\n', best_tag, best_score * 100);
fprintf('  Reference   : Hist-LBP standalone ? 88.00%%\n');


% =========================================================================
% FUNCTIONS
% =========================================================================

function best_C = find_best_C(X_raw, y, fold_ids, c_grid)
    best_C    = c_grid(1);
    best_score = -1;
    inner_folds = unique(fold_ids);
    for ci = 1:length(c_grid)
        C = c_grid(ci);
        scores = zeros(1, length(inner_folds));
        for fi = 1:length(inner_folds)
            f   = inner_folds(fi);
            Xtr = X_raw(fold_ids ~= f, :);
            Xva = X_raw(fold_ids == f, :);
            ytr = y(fold_ids ~= f);
            yva = y(fold_ids == f);
            [Xtr_n, Xva_n] = power_normalization(Xtr, Xva);
            mdl = fitcsvm(Xtr_n, ytr, 'KernelFunction', 'linear', ...
                          'BoxConstraint', C, 'Standardize', false, ...
                          'ClassNames', [0, 1]);
            preds    = predict(mdl, Xva_n);
            scores(fi) = mean(preds == yva);
        end
        if mean(scores) > best_score
            best_score = mean(scores);
            best_C     = C;
        end
    end
end

function [X_train_n, X_test_n] = power_normalization(X_train, X_test)
    X_train = sqrt(abs(X_train));
    X_test  = sqrt(abs(X_test));
    mu       = mean(X_train, 1);
    X_train_n = X_train - mu;
    X_test_n  = X_test  - mu;
end