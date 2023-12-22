function [Ind, F, G, obj]=EUFS_v1(X,K,lambda, alpha, maxiter)
% min |X-FG^T|_21 + lambda*Tr(G^TLG) + alpha*|F|_21  s.t. G>=0, G^TG=I
% Note: this function implements EUFS, but the notation and dimension is
% differnt from what describe in the paper
% Input:
%   X: data matrix, p by n, each column is a data sample
%   K: number of clusters
%   lambda: parameter used to control the contribution of the graph regularizer
%   alpha: parameter used to control the sparsity of F
%   maxiter: maximal number of iterations
% Output:
%   Ind: feature index ranked by the importance of features in descending
%        order, to select k features, simply use the top k index, i.e., Ind(1:k) 
%   F: latent feature matrix, p by K
%   G: pseudo cluster indicator, K by n
%   obj: the value of objective function, due to the nature of ADMM, obj
%       may fluctuate a little



p=size(X,1);
n=size(X,2);

% This part of code constructs the Laplacian Graph0
options = [];
%options.Metric = 'Euclidean';
options.NeighborMode = 'KNN';
options.k = 5;
%options.WeightMode = 'Binary';
options.WeightMode = 'HeatKernel';
W = constructW(X',options);
D = diag(sum(W,1));
L=D-W;

% This part of code initialize the F and G of the data matrix

 idx_G=kmeans(X',K,'Emptyaction','drop');
 G=zeros(n,K);
 for j=1:length(idx_G)
     G(j,idx_G(j))=1;
 end
 G=G+0.2;   % numerical stability purpose
%     
 idx_F=kmeans(X,K,'Emptyaction','drop');
 F=zeros(p,K);
 for i=1:length(idx_F)
     F(i,idx_F(i))=1;
 end
 F=F+0.2;

% This par of code initialize the gamma and sigma 
mu=1e-3;
pho=1.1;
eps=1e-4;
gamma=zeros(p,n);
sigma=zeros(size(G));
%maxiter=1000;
k=1;
obj = zeros(maxiter,1);

while k<=maxiter
    % This part minimizes E
    temp1=X-F*G'+1/mu*gamma;
    E = solve_l1l2(temp1, 1/mu);   %[E]=L21_solver(temp1,1/mu);
    
    % This part minimizes F
    %F=(X-E+1/mu*gamma)*G*(G'*G)^(-1);          %%%%%%%%%%%%%%%%%%%%%%%   F=(X-E+1/mu*gamma)*G;
    F = solve_l2l1((X-E+1/mu*gamma)*G, alpha/2);
    
    % This part minimizes H
    temp2=G+1/mu*sigma-lambda/mu*(G'*L)'; 
    [H]=nonneg_L2(temp2);
    
    % This part minimizes G
    %temp3=H-1/mu*sigma+lambda/mu*L*H+(F'*(X-E+1/mu*gamma))';
    temp3=H-1/mu*sigma-lambda/mu*L*H+(F'*(X-E+1/mu*gamma))';
    [U,S,V]=svd(temp3,0);
    G=U*V';
    
    % update the parameters
    gamma=gamma+mu*(X-F*G'-E);
    sigma=sigma+mu*(G-H);
    mu=pho*mu;
    
    obj(k) = RMNMF_FS_obj(X, F, G, L, lambda, alpha);
    fprintf('Iter %d\tobj=%f\n',k,obj(k));
    k=k+1;
end


% for i=1:size(G,1)
%     [temp1,temp2]=max(G(i,:));
%     idx(i)=temp2;
% end

temp4 = sqrt(sum(F.*F,2));
[~, Ind] = sort(temp4, 'descend');            


end




function [Obj] = RMNMF_FS_obj(X, F, G, L, lambda, alpha)
temp = X-F*G';
Obj = sum(sqrt(sum(temp.*temp, 1))) + lambda*trace(G'*L*G) + alpha*sum(sqrt(sum(F.*F,2)));
end

function [H]=nonneg_L2(temp)
% This function returns the max of temp elements and 0.

[n,p]=size(temp);
for i=1:n
    for j=1:p
        H(i,j)=max(temp(i,j),0);
    end
end

end


function [E] = solve_l1l2(W, lambda)
n = size(W,2);
E = W;
for i = 1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end
end

function [E] = solve_l2l1(W, lambda)
n = size(W,1);
E = W;
for i = 1:n
    E(i,:) = solve_l2(W(i,:),lambda);
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
end

