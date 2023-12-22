clc;
clear;
addpath(genpath('.\'));

% Define parameters
% lambda1 = 0.1;
% lambda2 = 0.1;
% lambda3 = 1;
% lambda4 = 0.0001;
lambda =1;
lambda1 = 0.5;
alpha = 0.5;
maxiter = 100;
%K = 10;


%alpha = input('Enter the alpha value: ');
Q = 'shannon_entropy_selected';
%q = 'compute_entropy';
num_folds = 5;  % Number of folds for cross-validation

% Load Xset
%X_sets = ["dna", "procancer"]; 
%X_sets = ["dna", "Yale1"]; 
%X_sets=["dna","cnscancer"];
%X_sets = ["dna","sport"];
%X_sets = ["dna","sport"];% Add more Xsets if needed
%X_sets = ["dna","endocrinecancer"];
X_sets = ["TOX-171", "ALLAML"]; 
 X = X_sets(2);
load(strcat(X, ".mat"));
K = length(unique(Y));

% Initialize array to store NMI values
nmi_values = zeros(num_folds, 1);

ARI_values = zeros(num_folds, 1);
% Perform k-fold cross-validation
cv = cvpartition(size(X, 1), 'KFold', num_folds);

for fold = 1:num_folds
    train_idx = training(cv, fold);
    test_idx = test(cv, fold);

    % Split X into training and testing sets
    X_train = X(train_idx, :);
    Y_train = Y(train_idx, :);

    X_test = X(test_idx, :);
    Y_test = Y(test_idx, :);



    % Feature selection using dufs_L
    Ind = EUFS_V2(X_train, size(Y_train, 2), Q, lambda, lambda1, alpha, maxiter);
   %Ind = EUFS_V2(X_train, K,Q,lambda,lambda1, alpha, maxiter);
   
   
 
    % Extract selected features
   % selected_features = X_train(:, Ind);
   
    % Extract selected features
   num_selected_features = min(K, size(X_train, 2));  % Ensure not to exceed the number of features
    Ind_valid = Ind(1:num_selected_features);  % Use only the valid indices within the available features
    
    selected_features = X_train(:, Ind_valid);

    % Cluster the X using k-means
    num_clusters = size(Y_train, 2);  % Number of clusters equals the number of classes
    [~, C] = kmeans(selected_features, num_clusters);

    % Assign X points to clusters
    [~, predicted_labels] = pdist2(C, selected_features, 'euclidean', 'Smallest', 1);

    
 

    % Evaluate clustering performance using NMI
    true_labels = vec2ind(Y_train');
    NMI_value = calculate_NMI(true_labels, predicted_labels);
    
    conf_matrix = confusionmat(true_labels, predicted_labels);
%disp(conf_matrix);

ARI = calculate_ARI(conf_matrix);
fprintf('Adjusted Rand Index: %.2f\n', ARI);



    % Store NMI value
    nmi_values(fold) = NMI_value;

    % Display NMI for each fold
    fprintf('Fold %d - NMI: %.2f\n', fold, NMI_value);
    
    
     
end


% Calculate mean and standard deviation of NMI values
mean_ARI = mean(ARI);
std_ARI = std(ARI);
% Display mean and standard deviation in the desired format
fprintf('\nMean ARI: %.2f ± %.2f\n', mean_ARI, std_ARI);


% Calculate mean and standard deviation of NMI values
mean_nmi = mean(nmi_values);
std_nmi = std(nmi_values);
% Display mean and standard deviation in the desired format
fprintf('\nMean NMI: %.2f ± %.2f\n', mean_nmi, std_nmi);






% Function to calculate Normalized Mutual Information (NMI)
%  function NMI = calculate_NMI(true_labels, predicted_labels)
%       true_labels = true_labels(:)';
%       predicted_labels = predicted_labels(:)';
% % %     
% % %     % Ensure true_labels and predicted_labels have the same length
%       min_len = min(length(true_labels), length(predicted_labels));
%       true_labels = true_labels(1:min_len);
%       predicted_labels = predicted_labels(1:min_len);
% % % 
%       confusion_matrix = confusionmat(true_labels, predicted_labels);
%       P = confusion_matrix / sum(confusion_matrix(:), 'omitnan');
% % %     
%       H_true = -sum(P(1, :) .* log2(P(1, :) + eps), 'omitnan');
%       H_pred = -sum(P(:, 1) .* log2(P(:, 1) + eps), 'omitnan');
%       H_joint = -sum(P(:) .* log2(P(:) + eps), 'omitnan');
% %      
%      NMI = (H_true + H_pred) / H_joint;
%   %NMI = nmi(true_labels, predicted_labels);
%  end


% function NMI = calculate_NMI(true_labels, predicted_labels)
%     true_labels = true_labels(:)';
%     predicted_labels = predicted_labels(:)';
% 
%     % Ensure true_labels and predicted_labels have the same length
%     min_len = min(length(true_labels), length(predicted_labels));
%     true_labels = true_labels(1:min_len);
%     predicted_labels = predicted_labels(1:min_len);
% 
%     confusion_matrix = confusionmat(true_labels, predicted_labels);
%     P = confusion_matrix / sum(confusion_matrix(:), 'omitnan');
% 
%     H_true = -sum(P(1, :) .* log2(P(1, :) + eps), 'omitnan');
%     H_pred = -sum(P(:, 1) .* log2(P(:, 1) + eps), 'omitnan');
%     H_joint = -sum(P(:) .* log2(P(:) + eps), 'omitnan');
% 
%     % Ensure the denominator is not zero
%     if H_joint == 0
%         NMI = 0;
%     else
%         NMI = (H_true + H_pred) / H_joint;
%     end
% end




function NMI = calculate_NMI(true_labels, predicted_labels)
    true_labels = true_labels(:)';
    predicted_labels = predicted_labels(:)';

    % Ensure true_labels and predicted_labels have the same length
    min_len = min(length(true_labels), length(predicted_labels));
    true_labels = true_labels(1:min_len);
    predicted_labels = predicted_labels(1:min_len);

    confusion_matrix = confusionmat(true_labels, predicted_labels);
    P = confusion_matrix / sum(confusion_matrix(:), 'omitnan');

    H_true = -sum(P(1, :) .* log2(P(1, :) + eps), 'omitnan');
    H_pred = -sum(P(:, 1) .* log2(P(:, 1) + eps), 'omitnan');
    H_joint = -sum(P(:) .* log2(P(:) + eps), 'omitnan');

    % Ensure the denominator is not zero
    if H_joint == 0
        NMI = 0;
    else
        NMI = (H_true + H_pred) / H_joint;
    end
    
    % Make sure NMI is in the range [0, 1]
    NMI = max(0, min(NMI, 1));
end

function ARI = calculate_ARI(conf_matrix)
    % Step 1: Calculate Rand Index (RI)
    TP_plus_TN = sum(sum(conf_matrix .* (conf_matrix - 1) / 2));
    all_pairs = nchoosek(sum(conf_matrix(:)), 2);
    RI = TP_plus_TN / all_pairs;

    % Step 2: Calculate Adjusted Rand Index (ARI)
    expected_RI = sum(sum(conf_matrix, 2) .* sum(conf_matrix, 1) / all_pairs);
    max_RI = (sum(sum(conf_matrix, 2).^2) + sum(sum(conf_matrix, 1).^2)) / (2 * all_pairs);
    ARI = (RI - expected_RI) / (max_RI - expected_RI);

    % Ensure ARI is in the range [-1, 1]
    ARI = max(-1, min(ARI, 1));
end




