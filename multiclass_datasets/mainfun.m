% Semi-Supervised label distribution learning via projection graph embedding
%
% For now, instead of LDL datasets, we are using Multi-label datasets
%
% SSLDL
% Transductive first
% C:\Users\Sanjay\Documents\SAU\Multi-label\Code\LDL\SemiSupervised

%addpath(genpath('.'));
clc
addpath('D:\mamta\data\multiclass_datasets');


fprintf('* Run SSLDL for multi-label learning in semi-supervised setting *\n');

%Parameters set as per guidance in the paper: Sec 4.3
param.gamma = 10^2;   % {10^1, 10^2, 10^3}
param.k = 50;         % 50 or 100      
param.t = 20;         % iteration
param.lambda = 10^0;  %{10^-1,...,10^2}
%param.c = 0.8 * num_of_features; % 80 percent of the original input space dimension

param.cv_num = 5;

%Later, Use LDL datasets: http://palm.seu.edu.cn/xgeng/LDL/index.htm
%multi-label for now
%datasets={'CAL500.mat' , 'birds.mat', 'genbase.mat', 'medical.mat', 'Image.mat'};
datasets={'satimage_multiclass.mat'};


for dc=1:numel(datasets)
    load(datasets{dc});
    fprintf("Current Dataset: %s\n", datasets{dc});

%Merge defaults and split using CV.
if exist('train_data','var')==1
    data    = [train_data; test_data];
    target  = [train_target, test_target];
end
clear train_data test_data train_target test_target

%start with multi-label, we will change it to LDL later
target(target == -1) = 0;

%Normalize Row wise. Change it to normal standard scalar later.
data      = double (data);
num_data  = size(data, 1);
temp_data = data + eps;
%Instead of row wise normalization(current), consider standard one. Please.
temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
if sum(sum(isnan(temp_data)))>0
    temp_data = data+eps;
    temp_data = temp_data./repmat(sqrt(sum(temp_data.^2,2)),1,size(temp_data,2));
end

%TBD: Revisit
%temp_data = [temp_data,ones(num_data, 1)]; %Add a column of 1s for x^0 to the data/ for b in y=ax+b

rng(42); %Answer to life and everything, Hitchhiker's
randorder = randperm(num_data);
cvResult  = zeros(16,param.cv_num); %Metrics Store

%Using Cross validation for basically circling through the data and
%creating 1:4 semi-supervised splits.
for j = 1:param.cv_num
    fprintf('- Datasets: %s, Cross Validation - %d/%d\n', datasets{dc}, j, param.cv_num);

    %% Only 1/cv_num fraction of labels known for semi-supervised learning
    [X_l, X_u, D_kl, D_ku] = generateSemiSupervisedSplit(temp_data, target', randorder,j, param.cv_num);

    %X is nxm and D is nxq respectively. 
    %Change to X -> mxn and D-> qxn (to match the dimension convention in paper.
    %D_kl and D_ku contains ground truth label information.
    X_l = X_l'; X_u = X_u'; D_kl = D_kl'; D_ku = D_ku';

    D_l = zeros(size(D_kl)); %initial prediction all 0, We don't have to do this.
    D_u = zeros(size(D_ku)); %initial prediction all 0

    X = [X_l, X_u]; 
    D = [D_l, D_u];

    [m, n] = size(X);
    [~, nl] = size(X_l); %nl: no of labeled instances, n-nl: no of unlabelled instances
    param.c = ceil(0.8 * m);
    
    %initialize S
    sigma2 = 1; k=50; %fix sigma^2 later
    Sall = exp(-squareform(pdist(X')).^2/(2*sigma2)); %Symmetric
    [~, neighbor_idx] = maxk(Sall - eye(size(Sall,1)), k); %column contains neighbor indexes
    
    %Set similarity with non-neighbors to 0
    S = zeros(size(Sall));
    for ii=1:size(Sall,1)
        %columns contain neighbors
        S(neighbor_idx(:,ii), ii) = Sall(neighbor_idx(:,ii), ii); 
    end
        
    D = SSLDL(X, S, param, X_l, X_u, D_l, D_u, D_kl, neighbor_idx);
 
    %% Prediction and evaluation
    %Use thresholding to convert D into 0 and 1's 
    Outputs = D;
    Pre_Labels = double(Outputs>=0.5); %Tuning etc later

    fprintf('-- Evaluation\n');
    tmpResult = EvaluationAll(Pre_Labels,Outputs, [D_kl, D_ku]);
    cvResult(:,j) = cvResult(:,j) + tmpResult;
end

Avg_Result      = zeros(16,2);
Avg_Result(:,1) = mean(cvResult,2);
Avg_Result(:,2) = std(cvResult,1,2);
model_SSLDL.avgResult = Avg_Result;
result    = Avg_Result;
model_SSLDL.cvResult  = cvResult;
PrintResults(Avg_Result);
end

function [V, Theta] = computeV(X, Ls, param, Theta)
    %On first call, S will be default init and Theta will be identity
    %V mxc matrix, Theta: mxm matrix
    
    %WARNING!!!: Ensure the eigenpairs contain real values only.
    %PROBLEM AREA: EVAL for some datasets are complex and large
    [evec, eval] = eig(param.gamma * X * Ls' * Ls * X' + param.lambda * Theta, 'vector');
    [~, ind] = sort(eval); % verify inc/dec
    evec = evec(:, ind);
    V = evec(:, 1:param.c); % take only c eigenvectors corresponding to the c smallest eigen values.
    %updated theta, l{2,1}-norm
    Theta =  diag(2*sqrt(sum(V.^2, 2)+eps)); %verify
end

function [S] = computeS(X, D, V, S, k, n_idx, gamma)
    %n_idx contains neighbors indices
    [~, n] = size(D);
    %S = zeros(n, n); %Nooooo, 
    for ii=1:n
        Hdi = (repmat(D(:,ii), 1, k) - D(:, n_idx(:,ii)))'; %di - din(i)
        Hxi = (repmat(V'*X(:,ii), 1, k) - V'*X(:,n_idx(:,ii)))'; %vi'xi - vi'xn(i)
        Bdi = Hdi * Hdi';
        Bxi = Hxi * Hxi';
        
        %All preparations done.
        %Now apply Quadprog here and solve for Si.
        %The dependence with older S is indirect and discomforting
        H = 2* (Bdi + gamma * Bxi);
        f = []; A = []; b = []; Aeq = ones(1,k); beq=[1]; lb = zeros(1,k); ub=ones(1,k);
        
        %solve for Si and spread the values to the neighbor locations
        %columnwise.  Old value of S are being overwritten instead of being updated???
        S(n_idx(:, ii), ii) = quadprog(2*H, f, A, b, Aeq, beq, lb, ub);     
    end
end

%D = computeD(X, S, param, Theta, X_l, X_u, D_l, D_u, D_kl, neighbor_idx);
function [D] = SSLDL(X, S, param, X_l, X_u, D_l, D_u, D_kl, n_idx)
    [m, nl] = size(X_l);
    [~, nu] = size(X_u); %nl: no of labeled instances, n-nl: no of unlabelled instances

    Theta = eye(m, m);
    iter = 0;
    while iter < 20  
        %Update V
        Ds =  diag(sum((S+S')/2, 2));
        Ls = Ds - (S+S')/2;
        [V, Theta] = computeV(X, Ls, param, Theta);

        %Update D
        Lll = Ls(1:nl, 1:nl);
        Llu = Ls(1:nl, nl+1:nl+nu);
        Lul = Ls(nl+1:nl+nu, 1:nl);
        Luu = Ls(nl+1:nl+nu, nl+1:nl+nu);
        D_u = -D_kl * Llu * pinv(Luu); %Fix dimension.
        %Fill D
        D = [D_kl, D_u];
        
        %Update S
        
        S = computeS(X, D, V, S, param.k, n_idx, param.gamma);
        
        iter = iter + 1;
    end
    D = [D_kl, D_u];
end