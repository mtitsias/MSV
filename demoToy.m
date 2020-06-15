

%clear all;
%close all;

randn('seed',0);
rand('seed',0);
outdir = 'diagrams/';

% LOAD the data 
load toy1.mat;
N = 2;   % number of log return sequences
T = 200; % number of time instances 
K = 2;   % number of latent factors 
% END LOADING the data 


% USER defined options
%
% -- Factor model or just simple MSV -- 
%    yes: factor MSV
%    no:  simple MSV  
model.useFactorModel = 'yes'; 
%
% -- Sample factors by Gibbs or not --
%    (this option has *no effect* if model.useFactorModel = 'no')
%      yes: it samples the factors by Gibbs (expensive, but exact) 
%      no:  it samples the factors by auxiliary Langevin (faster)
model.sampleFactorsByGibbs = 'no'; 
%
% -- Diagonal Sigmat matrix or not (i.e. indepedent factors or not) --
%      yes: the Sigmat matrices are all diagonal (angles are zero) 
%      no:  the Sigmat matrices have free form (angles are inferred)
model.diagonalSigmat = 'no';
%
% -- Exchangeable prior for the phis or just simple independent Gaussian with very
%    large variance
%    yes: exchangeable with normal-inverse gamma hyerprior 
%    no:  just a simple broad Gaussian 
model.exchangeablePriorphi = 'no';
%
% -- Single noise variance sigma2 in the final factor model 
%    or diagonal covariance (this option has been after the recent (2017) reviews)
%    yes:  
%    no: 
model.diagonalFinalLikelihoodNoise = 'yes';


if strcmp(model.useFactorModel, 'no') 
    K = N; 
end    

% create the Givens set
Givset = [];  % the indices 
for i=1:K
   for j=i+1:K
       Givset = [Givset; i j];
   end
end
tildeK = size(Givset,1);



% GENERATE TOY DATA BY SAMPLING FROM THE MSV MODEL
phi_h = 0.99*ones(1,K);
h_0 = 3*linspace(-0.5*K,0.5*K, K);
sigma2_h = 0.001*ones(1, K);
if toy == 1
   phi_delta = -0.99*ones(1, tildeK);
else
   phi_delta = 0.99*ones(1, tildeK);
end
delta_0 = zeros(1, tildeK);
sigma2_delta = 0.001*ones(1, tildeK); 
% GENERATE THE ARTIFICIAL DATA BY APPLYING THE FACTOR MODEL
[Ft, hs, deltas, Sigma] = generateToy(K, T, h_0, phi_h, sigma2_h, delta_0, phi_delta, sigma2_delta);
Ftoriginal = Ft; 
sigma2 = [0.01 0.01]; 
Y = Ft + bsxfun(@times, sqrt(sigma2)', randn(N,T));
Ytrue = Y;

 L = ones(K,K);

% Add some missing values
probNan = 0.1;  
for t=1:T
    r = rand(N,1);
    r = find(r<=probNan); 
    Y(r, t) = NaN; 
end


% START CREATING THE MODEL STRUCTURE
model.N = N;
model.T = T;
model.K = K; 
model.Y = Y;
model.Givset = Givset;
model.tildeK = size(Givset,1);

% PARAMETER INITIALIZATION FOR THE MCMC
model.deltas = zeros(model.tildeK, T); 
model.omegas = (0.5*pi)*( (exp(model.deltas)-1)./(exp(model.deltas) + 1));
model.hs = repmat(zeros(K,1), 1, T);
model.lambdas = exp(model.hs);


% HYPERPARAMETER INITIALIZATION FOR MCMC
ind =  ~isnan(model.Y(:)); 
if strcmp(model.diagonalFinalLikelihoodNoise, 'yes')
   model.sigma2 = 0.01*var(model.Y(ind))*ones(1,N); 
else
   model.sigma2 = 0.01*var(model.Y(ind)); 
end
model.L = L;
model.Weights = randn(N,K);
model.sigma2weights = 2;
model.Ft = zeros(model.K, model.T);
%model.FFt = Ft;
model.phi_h = zeros(1 ,K);
model.tildephi_h = log((1 + model.phi_h)./(1 - model.phi_h));
model.h_0 = zeros(1 ,K);
model.sigma2_h = ones(1, K);
model.phi_delta = zeros(1, tildeK);
model.tildephi_delta = log((1 + model.phi_delta)./(1 - model.phi_delta));
model.delta_0 = zeros(1, tildeK);
model.sigma2_delta = ones(1, tildeK); 


% PRIOR OVER PHIS 
if strcmp(model.exchangeablePriorphi, 'yes') 
model.priorPhi_h.type = 'logmarginalizedNormalGam'; 
model.priorPhi_h.mu0 = 0;
model.priorPhi_h.k0 = 1;
model.priorPhi_h.alpha0 = 1;
model.priorPhi_h.beta0 = 1; 
model.priorPhi_delta.type = 'logmarginalizedNormalGam'; 
model.priorPhi_delta.mu0 = 0;
model.priorPhi_delta.k0 = 1;
model.priorPhi_delta.alpha0 = 1;
model.priorPhi_delta.beta0 = 1; 
else
model.priorPhi_h.type = 'logNormal'; 
model.priorPhi_h.mu0 = 0;
model.priorPhi_h.s2 = 100;
model.priorPhi_delta.type = 'logNormal'; 
model.priorPhi_delta.mu0 = 0;
model.priorPhi_delta.s2 = 100;
end
model.priorSigma2_h.sigmar = 5; 
model.priorSigma2_h.Ssigma = 0.01*model.priorSigma2_h.sigmar;  
model.priorSigma2_delta.sigmar = 5;
model.priorSigma2_delta.Ssigma = 0.01*model.priorSigma2_delta.sigmar;  

% INVERSE GAMMA PRIOR OVER THE LIKELIHOOD NOISE VARIANCE
model.priorSigma2.type = 'invgamma';  
model.priorSigma2.alpha0 = 0.001;
model.priorSigma2.beta0 = 0.001;


% MCMC OPTIONS FOR BURNIN AND SAMPLING PHASES
mcmcoptions.adapt.T = 5;
mcmcoptions.adapt.Burnin = 0;
mcmcoptions.adapt.StoreEvery = 1;
mcmcoptions.adapt.disp = 1;
mcmcoptions.adapt.minAdapIters = 4;
mcmcoptions.train.T = 3;
mcmcoptions.train.Burnin = 0;
mcmcoptions.train.StoreEvery = 1;


% HERE WE RUN THE MCMC ALGORITHM FIRST TO ADAPT THE PROPOSAL AND THEN TO
% COLLECT THE SAMPLES
Langevin = 1;
tic;
[model PropDist samples accRates] = mcmcAdapt(model, mcmcoptions.adapt, Langevin); 
% training/sample collection phase
elapsedAdapt=toc;
tic;
[model samples accRates] = mcmcTrain(model, PropDist, mcmcoptions.train, Langevin);
elapsedTrain = toc;


% PLACE THE SAMPLES IN THE ORIGINAL FORM
m = mean(samples.F);
sd = sqrt(var(samples.F));
samples.hs = zeros(size(model.hs,1), size(model.hs,2), size(samples.F,1));
samples.deltas = zeros(size(model.deltas,1), size(model.deltas,2), size(samples.F,1));
for s=1:size(samples.F,1)
  samples.hs(:,:,s) = reshape(samples.F(s,1:K*T), T, K)';
  samples.deltas(:,:,s) = reshape(samples.F(s,(K*T+1):end), T, tildeK)';
end
samples.mean_hs = reshape(m(1:K*T), T, K)';
samples.sd_hs = reshape(sd(1:K*T), T, K)';
samples.mean_deltas = reshape(m((K*T+1):end), T, tildeK)';
samples.sd_deltas = reshape(sd((K*T+1):end), T, tildeK)';


% COMPUTE MONTE CARLO AVERAGES FOR THE COVARIANCES ACROSS TIME
estimSigma = zeros(N,N,T);
S = size(samples.F,1);
for s = 1:S
  omegas = (0.5*pi)*( (exp( samples.deltas(:,:,s) )-1)./(exp(  samples.deltas(:,:,s) ) + 1));
  lambdas = exp(samples.hs(:,:,s)); 
  LW = model.L.*samples.Weights(:,:,s);
  for t=1:T
     G = eye(K);
     for k=1:size(Givset,1)
        Gtmp = givensmat(omegas(k, t), K, Givset(k,1), Givset(k,2));
        G = Gtmp*G;
     end
     estimSigma(:,:,t) = estimSigma(:,:,t) + LW*(G*diag(lambdas(:,t))*G')*LW';
  end
end
estimSigma = estimSigma/S; 
 
  
% PLOT THE COVARIANCES FOR EACH PAIR OF RETURNS TOGETHER WITH THE GROUND-TRUTH
FS = 24;
for kk1=1:N
for kk2=kk1+1:N    
   ind = [kk1 kk2];  % pair   
   
   % plot the ground-truth and the estimated covariance as ellipses
   figure;
   hold on; 
   cnt = 0;
   % plot the Gaussian ellipsoids 
   jj = 0;
   for j=1:2:size(Sigma,3)
      jj = jj + 1; 
      h = plot_gaussian_ellipsoid([2*jj cnt*3], Sigma(:,:,j) + diag(sigma2));
      
      if strcmp(model.diagonalFinalLikelihoodNoise, 'yes')
      h1 = plot_gaussian_ellipsoid([2*jj cnt*3], estimSigma(ind,ind,j) + diag(mean(samples.sigma2)));  
      else    
      h1 = plot_gaussian_ellipsoid([2*jj cnt*3], estimSigma(ind,ind,j) + mean(samples.sigma2)*eye(2));
      end
      set(h1,'color','r');
      if jj==10
        cnt = cnt - 4;  
        jj = 0;
      end
   end
   set(gca, 'YTickLabel', []);
   set(gca, 'XTickLabel', []);
   set(gca,'fontsize',FS);
   box on; 
   axis tight; 
   title(['Pair: (' num2str(ind(1))  ','  num2str(ind(2)) ')' ' -- blue are the ground-truth and red the estimated covariances' ], 'Fontsize',FS);
   print('-depsc', [outdir 'fig_toyMSVpair' num2str(ind(1)) '-' num2str(ind(2))]);
   cmd = sprintf('epstopdf %s', [outdir 'fig_toyMSVpair' num2str(ind(1)) '-' num2str(ind(2)) '.eps']);
   system(cmd);
end
end

estimWeights = sum(samples.Weights, 3)/S;

estimWeights

% print also the mask of missing values 
figure;
Mask = Y; 
Mask = Mask*0 + 1;
ind = isnan(Mask);
Mask(ind) = 0;
imagesc(Mask);
colormap('gray');
axis off;
print('-depsc2', '-r300', [outdir 'fig_toyMSV_nanMask']);
cmd = sprintf('epstopdf %s', [outdir 'fig_toyMSV_nanMask.eps']);
system(cmd);
  

%disp(['Mean estimated phi_h=' num2str(mean(samples.Phi_h)) ';     Ground-truth=' num2str(phi_h)]); 
%disp(['Mean estimated phi_delta=' num2str(mean(samples.Phi_delta)) ';     Ground-truth=' num2str(phi_delta)]); 
%
%disp(['Mean estimated h_0=' num2str(mean(samples.h_0)) ';     Ground-truth=' num2str(h_0) ]); 
%disp(['Mean estimated delta_0=' num2str(mean(samples.delta_0)) ';     Ground-truth=' num2str(delta_0) ]); 
%
%disp(['Mean estimated sigma2_h ' num2str(mean(samples.sigma2_h)) ';     Ground-truth ' num2str(sigma2_h)]); 
%disp(['Mean estimated sigma2_delta ' num2str(mean(samples.sigma2_delta)) ';     Ground-truth ' num2str(sigma2_delta)]);
