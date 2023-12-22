function ret = eufsM( data, class_size, L, QQ, lambda1, lambda2, lambda3,lambda4 )
% implemented by lim
% 2015, AAAI, embedded ufs
% laplacian matrix (RBF kernel) should be provided
% E = X-UV', Z = U
% each row of X is a sample
% |X-UV'|

[n, d] = size( data );
X = data;
m = class_size;

% mu = 10^-3;
% in paper, mu was set in the range of 10^-6 to 10^-3
mu = lambda3;
rho = 1.1;
mu_max = 10^10;
%  U = zeros( n, m );
%  V = zeros( d, m );
% Initialize U and V without using qr
% Initialize U and V without using qr
 U = randn(n, class_size(1));
 V = randn(d, class_size(1));
% 
% % Normalize U and V to unit length
 U = U ./ sqrt(sum(U.^2, 1));
 V = V ./ sqrt(sum(V.^2, 1));




Q = 0;
Y1 = 0;
Y2 = 0;
E = zeros(n, d);
K = 0;
ITER = 10;


disp(size(U));
disp(size(L));
disp(size(Y1));


for i=1:ITER
	Q = X - U*V' + (1/mu)*Y2;

	% update E
	for j=1:n
		qi_norm = sqrt(sum( Q(j, :).^2 ));
		if qi_norm > (1/mu)
			E(j, :) = (1-(1/(mu*qi_norm))) * Q(j, :);
		else
			E(j, :) = 0;
		end
	end

	K = (X-E+(1/mu)*Y2)' * U;
      DD=diag(1/norm(V));
      V= (eye(d)+ lambda1*DD+lambda4*QQ)*K; 
%    
	% update V
% 	for j=1:d
% 		ki_norm = sqrt(sum( K(j, :).^2 ));
% 		if ki_norm > (lambda1/mu)
% %             fprintf('%d %d\n', i, j);
% 			V(j, :) = (1-(lambda1/(mu*ki_norm))) * K(j, :);
% 		else
% 			V(j, :) = 0;
% 		end
% 	end

	%T = U - (1/mu)*Y1 - (lambda2/mu)*L*U ;
    T = U - (1/mu)*Y1 - (lambda2/mu)*L * U;

	Z = T;
	Z(Z<0) = 0;
	N = (1/mu) * Y1 + Z - lambda2 * L * Z + (X-E+(1/mu)*Y2) * V;
	[t1, ~, t2] = svd(N, 'econ');
	U = t1 * t2';
	Y1 = Y1 + mu*(Z-U);
	Y2 = Y2 + mu*(X - U*V' - E);
	mu = max([rho*mu, mu_max]);
end

[~, ret] = sort( sum(V.^2, 2), 'descend' );
% [ret] = sort( sum(V.^2, 2), 'descend' );
% ret = V;

end