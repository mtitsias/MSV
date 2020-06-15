function [Y, hs, deltas, Sigma] = generateToy(N, T, h_0, phi_h, sigma2_h, delta_0, phi_delta, sigma2_delta)


% create the Givens set
Givset = [];  % the indices 
for i=1:N
   for j=i+1:N
       Givset = [Givset; i j];
   end
end

tildeN = size(Givset,1);

% generate the hs
for i=1:N 
    k2 = sigma2_h(i)/(1 - phi_h(i)^2);
    hs(i,1) = h_0(i) + sqrt(k2)*randn;
    for t=2:T
       k0 = h_0(i) + phi_h(i)*(hs(i,t-1) - h_0(i));
       hs(i,t) = k0 + sqrt(sigma2_h(i))*randn;
    end
end    

% generate the deltas
for j=1:tildeN 
    k2 = sigma2_delta(j)/(1 - phi_delta(j)^2);
    deltas(j,1) = delta_0(j) + sqrt(k2)*randn;
    for t=2:T
       k0 = delta_0(j) + phi_delta(j)*(deltas(j,t-1) - delta_0(j));
       deltas(j,t) = k0 + sqrt(sigma2_delta(j))*randn;
    end
end

omegas = (0.5*pi)*( (exp(deltas)-1)./(exp(deltas) + 1));

Y = zeros(N,T);
for t=1:T
   G = eye(N);
   for k=1:size(Givset,1)
      Gtmp = givensmat(omegas(k, t), N, Givset(k,1), Givset(k,2));
      G = Gtmp*G;
   end
   Sigma(:,:,t) = G*diag(exp(hs(:,t)))*G';
   Y(:,t) = G*(exp(0.5*hs(:,t)).*randn(N,1));
end
