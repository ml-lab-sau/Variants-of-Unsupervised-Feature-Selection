% function ret = calqmi( X )
% % row of X is sample
% % q_{i, j} = I(f_i, f_j)
% % q_{i, i} = sum( q(i, :) )
% 
% d = size(X, 2);
% ret = zeros( d, d );
% ent_f = zeros( d, 1 );
% for i=1:d
%     ent_f( i, 1 ) = compute_shannon_entropy( X(:, i) );
% end
% for i=1:d
%     for j=i+1:d
%         ret(i, j) = ent_f(i, 1) + ent_f(j, 1) - compute_shannon_entropy( [X(:, i), X(:, j)] );
%     end
% end
% ret = ret + ret';
% for i=1:d
%     ret(i, i) = sum(ret(:, i));
% end
% 
% end

% function ret = calqmi(X)
% % Calculate the mutual information matrix between features
% 
% d = size(X, 2);
% ret = zeros(d, d);
% 
% for i = 1:d
%     for j = 1:d
%         if i == j
%             ret(i, j) = compute_shannon_entropy(X(:, i));
%         else
%             ret(i, j) = compute_mutual_information(X(:, i), X(:, j));
%         end
%     end
% end
% 
% end


%Remove comment from here

% function ret = calqmi(X)
%     % Calculate the mutual information matrix between features
% 
%     d = size(X, 2);
%     ret = zeros(d, d);
% 
%     for i = 1:d
%         for j = 1:d
%             if i == j
%                 ret(i, j) = compute_shannon_entropy(X(:, i));
%             else
%                 ret(i, j) = compute_mutual_information(X(:, i), X(:, j));
%             end
%         end
%     end
% 
% end
% 
% function entropy_value = compute_shannon_entropy(variable)
%     % Remove NaN values
%     variable(isnan(variable)) = [];
% 
%     % Calculate probability distribution
%     probabilities = histcounts(variable, 'Normalization', 'probability');
% 
%     % Calculate Shannon entropy
%     entropy_value = -sum(probabilities .* log2(probabilities + eps));
% end
% 
% 
% function mi_value = compute_mutual_information(X, Y)
%     % Calculate the mutual information between two variables X and Y
% 
%     % Remove NaN values
%     X(isnan(X)) = [];
%     Y(isnan(Y)) = [];
% 
%     % Joint probability distribution
%     joint_prob = histcounts2(X, Y, 'Normalization', 'probability');
% 
%     % Marginal probability distributions
%     prob_X = sum(joint_prob, 2);
%     prob_Y = sum(joint_prob, 1);
% 
%     % Calculate mutual information
%     mi_value = 0;
%     for i = 1:length(prob_X)
%         for j = 1:length(prob_Y)
%             if joint_prob(i, j) > 0
%                 mi_value = mi_value + joint_prob(i, j) * log2(joint_prob(i, j) / (prob_X(i) * prob_Y(j)));
%             end
%         end
%     end
% 
%     % Ensure non-negativity
%     mi_value = max(0, mi_value);
% end
% Remove upto here 


function ret = calqmi(X)
    % Calculate the mutual information matrix between features

    d = size(X, 2);
    ret = zeros(d, d);

    for i = 1:d
        for j = 1:d
            % Compute mutual information matrix based on Shannon's entropy
            ret(i, j) = compute_mutual_information_shannon(X(:, i), X(:, j));
        end
    end

end

function mi = compute_mutual_information_shannon(feature1, feature2)
    % Compute mutual information based on Shannon's entropy

    % Compute Shannon entropy for each feature
    entropy1 = compute_shannon_entropy(feature1);
    entropy2 = compute_shannon_entropy(feature2);

    % Compute joint entropy
    joint_entropy = compute_joint_entropy(feature1, feature2);

    % Calculate mutual information
    mi = entropy1 + entropy2 - joint_entropy;
end

function entropy = compute_shannon_entropy(X)
    % Compute Shannon entropy for a given feature

    % Your Shannon entropy calculation logic goes here
    % Compute Shannon entropy for a given feature

    % Get unique values and their counts
    [unique_values, ~, index] = unique(X);
    counts = accumarray(index, 1);

    % Normalize counts to get probabilities
    probabilities = counts / sum(counts);

    % Exclude zero probabilities to avoid log(0)
    probabilities = probabilities(probabilities > 0);

    % Compute Shannon entropy
    entropy = -sum(probabilities .* log2(probabilities));
end

    % Replace the following line with your Shannon entropy calculation
%     entropy = 0; % Placeholder, replace with actual calculation
% end

function joint_entropy = compute_joint_entropy(feature1, feature2)
    % Compute joint entropy for two features

    % Your joint entropy calculation logic goes here
    % ...
     % Compute joint entropy for two features

    % Combine the two features into a matrix
    joint_X = [feature1, feature2];

    % Get unique pairs of values and their counts
    [unique_pairs, ~, index] = unique(joint_X, 'rows');
    counts = accumarray(index, 1);

    % Normalize counts to get probabilities
    probabilities = counts / sum(counts);

    % Exclude zero probabilities to avoid log(0)
    probabilities = probabilities(probabilities > 0);

    % Compute joint entropy
    joint_entropy = -sum(probabilities .* log2(probabilities));
end

    % Replace the following line with your joint entropy calculation
%     joint_entropy = 0; % Placeholder, replace with actual calculation
% end


