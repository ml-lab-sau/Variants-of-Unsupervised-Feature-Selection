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
alpha = input('Enter the alpha value: ');
maxiter = 100;
K = 10;
%data = data.data;

q = 'compute entropy';
num_folds = 5;  % Number of folds for cross-validation

% Load dataset
data_sets = ["dna", "procancer"]; 
%data_sets = ["dna", "Yale1"]; 
%data_sets=["dna","cnscancer"];
%data_sets = ["dna","Prostate-GE"];
%data_sets = ["dna","sport"];% Add more datasets if needed
%data_sets = ["dna" , "endocrinecancer"];
 dataset = data_sets(2);
load(strcat(dataset, ".mat"));

% Initialize array to store NMI values
nmi_values = zeros(num_folds, 1);

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
   % Ind = EUFS_v1(data_train, size(target_train, 2), Q, lambda1, lambda2, lambda3, lambda4);
   Ind = EUFS_V3_Test_purpose(data_train, K,lambda,lambda1, alpha, maxiter);
    



    % Extract selected features
   % selected_features = data_train(:, Ind);
   
    % Extract selected features
   num_selected_features = min(K, size(data_train, 2));  % Ensure not to exceed the number of features
    Ind_valid = Ind(1:num_selected_features);  % Use only the valid indices within the available features
    
    %selected_features = data_train(:, Ind_valid);
    
    
     selected_features_train = data_train(:, Ind);
    selected_features_test = data_test(:, Ind);


    % Cluster the data using k-means
    num_clusters = size(target_train, 2);  % Number of clusters equals the number of classes
    [~, C] = kmeans(selected_features_train, num_clusters);

    % Assign data points to clusters
    [~, predicted_labels_train] = pdist2(C, selected_features_train, 'euclidean', 'Smallest', 1);
    
     % Assign data points to clusters for testing set
    [~, predicted_labels_test] = pdist2(C, selected_features_test, 'euclidean', 'Smallest', 1);

    
 

    % Evaluate clustering performance using NMI
    true_labels_train = vec2ind(target_train');
     accuracy_train = calculate_clustering_accuracy(true_labels_train, predicted_labels_train);

    % Calculate clustering accuracy for testing set
    true_labels_test = vec2ind(target_test');
    accuracy_test = calculate_clustering_accuracy(true_labels_test, predicted_labels_test);

    % Store clustering accuracy values
    accuracy_values(fold) = (sum(true_labels_train == predicted_labels_train) + sum(true_labels_test == predicted_labels_test)) / ...
        (length(true_labels_train) + length(true_labels_test));

    % Display clustering accuracy for each fold
    fprintf('Fold %d - Total Accuracy: %.2f%%\n', fold, accuracy_values(fold) * 100);
end

% Calculate mean and standard deviation of clustering accuracy values
mean_accuracy = mean(accuracy_values);
std_accuracy = std(accuracy_values);

% Display mean and standard deviation of clustering accuracy
fprintf('\nMean Clustering Accuracy: %.2f%% ± %.2f%%\n', mean_accuracy * 100, std_accuracy * 100);

% Function to calculate clustering accuracy
% Function to calculate clustering accuracy
function accuracy = calculate_clustering_accuracy(true_labels, predicted_labels)
    % Get unique labels from true labels and predicted labels
    unique_true_labels = unique(true_labels);
    unique_predicted_labels = unique(predicted_labels);

    % Create a mapping between true labels and cluster assignments
    label_mapping = containers.Map(unique_true_labels, 1:length(unique_true_labels));

    % Map true labels and predicted labels to integers
    true_labels_mapped = cell2mat(values(label_mapping, num2cell(true_labels)));
    
    % Only map predicted labels that are present in the true labels
    predicted_labels_mapped = zeros(size(predicted_labels));
    for i = 1:length(unique_predicted_labels)
        label = unique_predicted_labels(i);
        if isKey(label_mapping, label)
            predicted_labels_mapped(predicted_labels == label) = label_mapping(label);
        end
    end

    % Calculate clustering accuracy
    accuracy = sum(true_labels_mapped == predicted_labels_mapped) / length(true_labels_mapped);
end
