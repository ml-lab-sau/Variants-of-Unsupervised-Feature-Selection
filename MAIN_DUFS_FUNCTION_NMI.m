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

%K = length(unique(target));
%K = 2 ;


%alpha = input('Enter the alpha value: ');
%q = 'shannon_entropy_selected';
q = 'compute_entropy';
num_folds = 5;  % Number of folds for cross-validation

% Load dataset
%data_sets = ["dna", "procancer"]; 
%data_sets = ["dna", "Yale1"]; 
%data_sets=["dna","cnscancer"];
%data_sets = ["dna","sport"];
%data_sets = ["dna","sport"];% Add more datasets if needed
data_sets = ["dna","endocrinecancer"];
 data = data_sets(2);
load(strcat(data, ".mat"));
K = length(unique(target));

% Initialize array to store NMI values
nmi_values = zeros(num_folds, 1);

ARI_values = zeros(num_folds, 1);
% Perform k-fold cross-validation
cv = cvpartition(size(data, 1), 'KFold', num_folds);

for fold = 1:num_folds
    train_idx = training(cv, fold);
    test_idx = test(cv, fold);

    % Split data into training and testing sets
    data_train = data(train_idx, :);
    target_train = target(train_idx, :);

    data_test = data(test_idx, :);
    target_test = target(test_idx, :);



    % Feature selection using dufs_L
    Ind = EUFS_V2(data_train, size(target_train, 2), lambda, lambda1, alpha, maxiter);
   %Ind = EUFS_V3_Test_purpose(data_train, K,lambda,lambda1, alpha, maxiter);
   
   
 
    % Extract selected features
   % selected_features = data_train(:, Ind);
   
    % Extract selected features
   num_selected_features = min(K, size(data_train, 2));  % Ensure not to exceed the number of features
    Ind_valid = Ind(1:num_selected_features);  % Use only the valid indices within the available features
    
    selected_features = data_train(:, Ind_valid);

    % Cluster the data using k-means
    num_clusters = size(target_train, 2);  % Number of clusters equals the number of classes
    [~, C] = kmeans(selected_features, num_clusters);

    % Assign data points to clusters
    [~, predicted_labels] = pdist2(C, selected_features, 'euclidean', 'Smallest', 1);

    
 

    % Evaluate clustering performance using NMI
    true_labels = vec2ind(target_train');
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


% 
% 
% function z = nmi(true_labels, predicted_labels)
% % Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
% % Input:
% % x, y: two integer vector of the same length
% % Ouput:
% % z: normalized mutual information z=I(x,y)/sqrt(H(x)*H(y))
% % Written by Mo Chen (sth4nth@gmail.com).
% assert(numel(predicted_labels) == numel(true_labels));
% n = numel(predicted_labels);
% predicted_labels = reshape(predicted_labels,1,n);
% true_labels = reshape(true_labels,1,n);
% l = min(min(predicted_labels),min(true_labels));
% predicted_labels = predicted_labels-l+1;
% true_labels = true_labels-l+1;
% k = max(max(predicted_labels),max(true_labels));
% idx = 1:n;
% Mx = sparse(idx,predicted_labels,1,n,k,n);
% My = sparse(idx,true_labels,1,n,k,n);
% Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
% Hxy = -dot(Pxy,log2(Pxy));
% % hacking, to elimative the 0log0 issue
% Px = nonzeros(mean(Mx,1));
% Py = nonzeros(mean(My,1));
% % entropy of Py and Px
% Hx = -dot(Px,log2(Px));
% Hy = -dot(Py,log2(Py));
% % mutual information
% MI = Hx + Hy - Hxy;
% % normalized mutual information
% z = sqrt((MI/Hx)*(MI/Hy));
% z = max(0,z);
% 
% end

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




