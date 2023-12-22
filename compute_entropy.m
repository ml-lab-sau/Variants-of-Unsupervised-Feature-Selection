function entropy = compute_entropy(X, entropy_type, alpha)
    % Compute various entropy measures for a given feature

    switch entropy_type
        case 'shannon'
            entropy = compute_shannon_entropy(X);
        case 'renyi'
            entropy = compute_renyi_entropy(X, alpha);
        case 'collision'
            entropy = compute_collision_entropy(X);
        case 'min_entropy'
            entropy = compute_min_entropy(X, alpha);
        case 'hartley'
            entropy = compute_hartley_entropy(X);
        otherwise
            error('Invalid entropy type.');
    end
end

function entropy = compute_renyi_entropy(X, alpha)
    % Compute Rényi entropy for a given feature and alpha

    % Get unique values and their counts
    [unique_values, ~, index] = unique(X);
    counts = accumarray(index, 1);

    % Normalize counts to get probabilities
    probabilities = counts / sum(counts);

    % Compute Rényi entropy
    entropy = 1 / (1 - alpha) * log2(sum(probabilities.^alpha));
end

function entropy = compute_collision_entropy(X)
    % Compute Collision entropy for a given feature

    % Get unique values and their counts
    [unique_values, ~, index] = unique(X);
    counts = accumarray(index, 1);

    % Normalize counts to get probabilities
    probabilities = counts / sum(counts);

    % Compute Collision entropy
    entropy = -sum(probabilities .* log2(probabilities));
end

function entropy = compute_min_entropy(X, alpha)
    % Compute Min entropy for a given feature and alpha

    % Get unique values and their counts
    [unique_values, ~, index] = unique(X);
    counts = accumarray(index, 1);

    % Normalize counts to get probabilities
    probabilities = counts / sum(counts);

    % Compute Min entropy
    entropy = -log2(max(probabilities));
end

function entropy = compute_hartley_entropy(X)
    % Compute Hartley entropy for a given feature

    % Get unique values and their counts
    [unique_values, ~, index] = unique(X);
    counts = accumarray(index, 1);

    % Normalize counts to get probabilities
    probabilities = counts / sum(counts);

    % Compute Hartley entropy
    entropy = log2(length(unique_values));
end

% % Example for using Renyi entropy with user-defined alpha value
% entropy_type_renyi = 'renyi';
% alpha_renyi = input('Enter the alpha value for Renyi entropy: ');
% entropy_renyi = compute_entropy(X(:, i), entropy_type_renyi, alpha_renyi);
% 
% % Example for using Collision entropy with user-defined alpha value
% entropy_type_collision = 'collision';
% alpha_collision = input('Enter the alpha value for Collision entropy: ');
% entropy_collision = compute_entropy(X(:, i), entropy_type_collision, alpha_collision);
% 
% % Example for using Min entropy with user-defined alpha value
% entropy_type_min_entropy = 'min_entropy';
% alpha_min_entropy = input('Enter the alpha value for Min entropy: ');
% entropy_min_entropy = compute_entropy(X(:, i), entropy_type_min_entropy, alpha_min_entropy);
% 
% % Example for using Hartley entropy
% entropy_type_hartley = 'hartley';
% entropy_hartley = compute_entropy(X(:, i), entropy_type_hartley, []);
