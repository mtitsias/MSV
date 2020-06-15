function K = tridiagAR(T, phi, sigma2)
%

d = ((1+phi^2)/sigma2)*ones(T,1);

K = (-phi/sigma2)*ones(T,T);
K = triu(K,-1) - triu(K,2);
K = K + diag(d - diag(K));

K(1,1) = 1/sigma2;
K(T,T) = 1/sigma2;

K = sparse(K);

