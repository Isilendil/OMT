function  EOK()
% EObeta: Effect of beta
%--------------------------------------------------------------------------
% Input: 
%      dataset_name, name of the dataset, e.g. 'books_dvd'
%
% Output: a figure for the effect of parameter C on the dataset
%--------------------------------------------------------------------------

%load dataset
load(sprintf('../data/%s', 'books_dvd'));
[n,d]       = size(data);
% set parameters

dataset_name = 'books_dvd';

%% set parameters: 'sigma'( kernel width) and 't_tick'(step size for plotting figures)
switch (dataset_name)
    case 'usenet2'
        options.sigma =  4;
        options.sigma2 = 8;
        options.t_tick= 100;
    otherwise % default: sigma = 10
        options.sigma = 4;
        options.sigma2 = 8;
        options.t_tick= round(length(ID_new)/15);
end
%%  
m = length(ID_new);
options.Number_old=n-m;
options.beta = sqrt(m)/(sqrt(m)+sqrt(log(2)));
%ID_old = 1:n-m;
Y=data(1:n,1);
Y=full(Y);
X = data(1:n,2:d);


%% scale
MaxX=max(X,[],2);
MinX=min(X,[],2);
DifX=MaxX-MinX;
idx_DifNonZero=(DifX~=0);
DifX_2=ones(size(DifX));
DifX_2(idx_DifNonZero,:)=DifX(idx_DifNonZero,:);
X = bsxfun(@minus, X, MinX);
X = bsxfun(@rdivide, X , DifX_2);


P = sum(X.*X,2);
P = full(P);
disp('Pre-computing kernel matrix...');
K = exp(-(repmat(P',n,1) + repmat(P,1,n)- 2*X*X')/(2*options.sigma^2));
% K = X*X';

X2 = X(n-m+1:n,:);
Y2 = Y(n-m+1:n);
P2 = sum(X2.*X2,2);
P2 = full(P2);
K2 = exp(-(repmat(P2',m,1) + repmat(P2,1,m)- 2*X2*X2')/(2*options.sigma2^2));
% K2 = X2*X2';

%================
vector_K = 2 .^ [0:8];
options.eta2 = 1/10;
options.C = 1;
%================

for ix =1:length(vector_K),
     options.K   = vector_K(ix);
     
     %% learn the old classifier
     [h, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = avePA1_K_M(Y, K, options, ID_old);
     fprintf(1,'The old classifier has %f support vectors\n',length(h.SV));
     
     %% run experiments:
     for i=1:size(ID_new,1),
         fprintf(1,'running on the %d-th trial...\n',i);
         ID = ID_new(i, :);      
         
        
    %4. OTL-I
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = HomOTL1_K_M(Y,K,K2,options,ID,h);
    nSV_OTL(i) = length(classifier.SV1)+length(classifier.SV2);
    err_OTL(i) = err_count;
    time_OTL(i) = run_time;
    mistakes_list_OTL(i,:) = mistakes;
    SVs_OTL(i,:) = SVs;
    TMs_OTL(i,:) = TMs;
    
     %5. OTL-II
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = HomOTL3_K_M(Y,K,K2,options,ID,h);
    nSV_OTL2(i) = length(classifier.SV1)+length(classifier.SV2);
    err_OTL2(i) = err_count;
    time_OTL2(i) = run_time;
    mistakes_list_OTL2(i,:) = mistakes;
    SVs_OTL2(i,:) = SVs;
    TMs_OTL2(i,:) = TMs;
    
     end
    ERR_OTL(ix) = mean(err_OTL)/m*100;
    ERR_OTL2(ix)= mean(err_OTL2)/m*100;

end

 %% print and plot results
 mistakes_idx = [0:8];
figure
plot(mistakes_idx, ERR_OTL,'ro-');
hold on
plot(mistakes_idx, ERR_OTL2,'rx-');
legend('HomOTL-I','HomOTL-II');
xlabel('log_2(K)');
ylabel('Average rate of mistakes')
grid


