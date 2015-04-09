function  EOK()
% EObeta: Effect of beta
%--------------------------------------------------------------------------
% Input: 
%      dataset_name, name of the dataset, e.g. 'books_dvd'
%
% Output: a figure for the effect of parameter C on the dataset
%--------------------------------------------------------------------------


%load dataset
load(sprintf('data/%s','vehicle-tree'));

%=========================================
load(sprintf('data/%s','vehicle-tree-similarity'));
P_it = P_ti';
%=========================================


ID_old = randperm(300);
for i = 1 : 20
    ID_new(i,:) = randperm(300)+300;
end

Y = image_gnd;
Y = full(Y);
X = image_fea;

[n,d] = size(image_fea);
% set parameters
options.C   = 5;
options.sigma = 4;
options.sigma2 = 8;
options.t_tick = round(size(ID_new,2)/10);


%%
m = size(ID_new,2);
options.beta=sqrt(m)/(sqrt(m)+sqrt(log(2)));
options.Number_old=n-m;
%ID_old = 1:n-m;



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
vector_K = 2 .^ [0:6];
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
         
        
    %1. MT-I
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = Multitransfer1(Y,K,K2,P_it,text_gnd,options,ID,h);
    nSV_MT1(i) = length(classifier.SV1)+length(classifier.SV2);
    err_MT1(i) = err_count;
    time_MT1(i) = run_time;
    mistakes_list_MT1(i,:) = mistakes;
    SVs_MT1(i,:) = SVs;
    TMs_MT1(i,:) = TMs;
    
     %2. MT-II
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = Multitransfer2(Y,K,K2,P_it,text_gnd,options,ID,h);
    nSV_MT2(i) = length(classifier.SV1)+length(classifier.SV2);
    err_MT2(i) = err_count;
    time_MT2(i) = run_time;
    mistakes_list_MT2(i,:) = mistakes;
    SVs_MT2(i,:) = SVs;
    TMs_MT2(i,:) = TMs;
    
     end
    ERR_MT1(ix) = mean(err_MT1)/m*100;
    ERR_MT2(ix)= mean(err_MT2)/m*100;

end

 %% print and plot results
 mistakes_idx = [0:6];
figure
plot(mistakes_idx, ERR_MT1,'ro-');
hold on
plot(mistakes_idx, ERR_MT2,'rx-');
legend('Multitransfer-I','Multitransfer-II');
xlabel('log_2(K)');
ylabel('Average rate of mistakes')
grid


