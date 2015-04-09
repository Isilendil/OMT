function  Experiment(data_file, similarity_file)
% Experiment: the main function used to compare all the online
% algorithms
%--------------------------------------------------------------------------
% Input:
%      dataset_name, name of the dataset, e.g. 'birds-food'
%
% Output:
%      a table containing the accuracies, the numbers of support vectors,
%      the running times of all the online learning algorithms on the
%      inputed datasets
%      a figure for the online average accuracies of all the online
%      learning algorithms
%      a figure for the online numbers of SVs of all the online learning
%      algorithms
%      a figure for the online running time of all the online learning
%      algorithms
%--------------------------------------------------------------------------

%load dataset
load(sprintf('data/%s', data_file));
%load(sprintf('%s','boats_toy'));
%load(sprintf('%s','flowers_tree'));
%load(sprintf('%s','vehicle_tree'));

%=========================================
load(sprintf('data/%s', similarity_file));
%load(sprintf('%s','boats_toy_sim'));
%load(sprintf('%s','flowers_tree_sim'));
%load(sprintf('%s','vehicle_tree_sim'));
P_it = P_ti';
%=========================================


ID_old = randperm(300);
for i = 1 : 100
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

%====================================
options.K = 10;
%====================================

%%
m = size(ID_new,2);
options.beta=sqrt(m)/(sqrt(m)+sqrt(log(2)));
options.Number_old=n-m;
%ID_old = 1:n-m;



% scale
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
%K = X*X';

X2 = X(n-m+1:n,:);
Y2 = Y(n-m+1:n);
P2 = sum(X2.*X2,2);
P2 = full(P2);
K2 = exp(-(repmat(P2',m,1) + repmat(P2,1,m)- 2*X2*X2')/(2*options.sigma2^2));
% K2 = X2*X2';

%% learn the old classifier
[h, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = avePA1_K_M(Y, K, options, ID_old);

fprintf(1,'The old classifier has %f support vectors\n',length(h.SV));
%% run experiments:
for i=1:size(ID_new,1),
    fprintf(1,'running on the %d-th trial...\n',i);
    ID = ID_new(i, :);
    
    %1. PA-I
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = PA1_K_M(Y2,K2,options,ID);
    nSV_PA1(i) = length(classifier.SV);
    err_PA1(i) = err_count;
    time_PA1(i) = run_time;
    mistakes_list_PA1(i,:) = mistakes;
    SVs_PA1(i,:) = SVs;
    TMs_PA1(i,:) = TMs;
    
    
    %2. PAIO
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = PAIO_K_M(Y,K,options,ID,h);
    nSV_PAI(i) = length(classifier.SV);
    err_PAI(i) = err_count;
    time_PAI(i) = run_time;
    mistakes_list_PAI(i,:) = mistakes;
    SVs_PAI(i,:) = SVs;
    TMs_PAI(i,:) = TMs;

    %3. HomOTL-I
    %[classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = Multitransfer11(Y,K,K2,P_it,text_gnd,options,ID,h);
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = HomOTL1_K_M(Y,K,K2,options,ID,h);
    nSV_OTL(i) = length(classifier.SV1)+length(classifier.SV2);
    err_OTL(i) = err_count;
    time_OTL(i) = run_time;
    mistakes_list_OTL(i,:) = mistakes;
    SVs_OTL(i,:) = SVs;
    TMs_OTL(i,:) = TMs;
    
    %4. HomOTL-II
    %[classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = Multitransfer21(Y,K,K2,P_it,text_gnd,options,ID,h);
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = HomOTL2_K_M(Y,K,K2,options,ID,h);
    nSV_OTL2(i) = length(classifier.SV1)+length(classifier.SV2);
    err_OTL2(i) = err_count;
    time_OTL2(i) = run_time;
    mistakes_list_OTL2(i,:) = mistakes;
    SVs_OTL2(i,:) = SVs;
    TMs_OTL2(i,:) = TMs;
    
    %5. Multitransfer-I
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = Multitransfer1(Y,K,K2,P_it,text_gnd,options,ID,h);
    nSV_MT1(i) = length(classifier.SV1)+length(classifier.SV2);
    err_MT1(i) = err_count;
    time_MT1(i) = run_time;
    mistakes_list_MT1(i,:) = mistakes;
    SVs_MT1(i,:) = SVs;
    TMs_MT1(i,:) = TMs;
    
    %6. Multitransfer-II
    [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = Multitransfer2(Y,K,K2,P_it,text_gnd,options,ID,h);
    nSV_MT2(i) = length(classifier.SV1)+length(classifier.SV2);
    err_MT2(i) = err_count;
    time_MT2(i) = run_time;
    mistakes_list_MT2(i,:) = mistakes;
    SVs_MT2(i,:) = SVs;
    TMs_MT2(i,:) = TMs;
    

end


stat_file = sprintf('stat/%s-stat', data_file);
save(stat_file, 'nSV_PA1', 'err_PA1', 'time_PA1', 'mistakes_list_PA1', 'SVs_PA1', 'TMs_PA1', 'nSV_PAI', 'err_PAI', 'time_PAI', 'mistakes_list_PAI', 'SVs_PAI', 'TMs_PAI', 'nSV_OTL', 'err_OTL', 'time_OTL', 'mistakes_list_OTL', 'SVs_OTL', 'TMs_OTL', 'nSV_OTL2', 'err_OTL2', 'time_OTL2', 'mistakes_list_OTL2', 'SVs_OTL2', 'TMs_OTL2', 'nSV_MT1', 'err_MT1', 'time_MT1', 'mistakes_list_MT1', 'SVs_MT1', 'TMs_MT1', 'nSV_MT2', 'err_MT2', 'time_MT2', 'mistakes_list_MT2', 'SVs_MT2', 'TMs_MT2');


%% print and plot results
figure
mean_mistakes_PA1 = mean(mistakes_list_PA1);
std_mistakes_PA1 = std(mistakes_list_PA1);
errorbar(mistakes_idx, mean_mistakes_PA1,std_mistakes_PA1,'g*-');
hold on
mean_mistakes_PAI = mean(mistakes_list_PAI);
std_mistakes_PAI = std(mistakes_list_PAI);
errorbar(mistakes_idx, mean_mistakes_PAI,std_mistakes_PAI,'g+-');
mean_mistakes_OTL = mean(mistakes_list_OTL);
std_mistakes_OTL = std(mistakes_list_OTL);
errorbar(mistakes_idx, mean_mistakes_OTL,std_mistakes_OTL,'bo-');
mean_mistakes_OTL2 = mean(mistakes_list_OTL2);
std_mistakes_OTL2 = std(mistakes_list_OTL2);
errorbar(mistakes_idx, mean_mistakes_OTL2,std_mistakes_OTL2,'bx-');
mean_mistakes_MT1 = mean(mistakes_list_MT1);
std_mistakes_MT1 = std(mistakes_list_MT1);
errorbar(mistakes_idx, mean_mistakes_MT1,std_mistakes_MT1,'ro-');
mean_mistakes_MT2 = mean(mistakes_list_MT2);
std_mistakes_MT2 = std(mistakes_list_MT2);
errorbar(mistakes_idx, mean_mistakes_MT2,std_mistakes_MT2,'rx-');
legend('PA', 'PAIO', 'HomOTL-I','HomOTL-II', 'Multitransfer1', 'Multitransfer2');
xlabel('Number of samples');
ylabel('Online average rate of mistakes')
grid
%{
pdf_file = sprintf('pdf/%s.pdf', data_file);
saveas(gcf, pdf_file);
close(figure(gcf));

figure
mean_SV_PA1 = mean(SVs_PA1);
plot(mistakes_idx, mean_SV_PA1,'g*-');
hold on
mean_SV_PAI = mean(SVs_PAI);
plot(mistakes_idx, mean_SV_PAI,'g+-');
mean_SV_OTL = mean(SVs_OTL);
plot(mistakes_idx, mean_SV_OTL,'bo-');
mean_SV_OTL2 = mean(SVs_OTL2);
plot(mistakes_idx, mean_SV_OTL2,'bx-');
mean_SV_MT1 = mean(SVs_MT1);
plot(mistakes_idx, mean_SV_MT1,'ro-');
mean_SV_MT2 = mean(SVs_MT2);
plot(mistakes_idx, mean_SV_MT2,'rx-');
legend('PA', 'PAIO', 'HomOTL-I','HomOTL-II', 'Multitransfer1', 'Multitransfer2', 'Location', 'Northwest');
xlabel('Number of samples');
ylabel('Online average number of support vectors')
grid

close(figure(gcf));

figure
mean_TM_PA1 = log(mean(TMs_PA1))/log(10);
plot(mistakes_idx, mean_TM_PA1,'g*-');
hold on
mean_TM_PAI = log(mean(TMs_PAI))/log(10);
plot(mistakes_idx, mean_TM_PAI,'g+-');
mean_TM_OTL = log(mean(TMs_OTL))/log(10);
plot(mistakes_idx, mean_TM_OTL,'bo-');
mean_TM_OTL2 = log(mean(TMs_OTL2))/log(10);
plot(mistakes_idx, mean_TM_OTL2,'bx-');
mean_TM_MT1 = log(mean(TMs_MT1))/log(10);
plot(mistakes_idx, mean_TM_MT1,'ro-');
mean_TM_MT2 = log(mean(TMs_MT2))/log(10);
plot(mistakes_idx, mean_TM_MT2,'rx-');
legend('PA', 'PAIO', 'HomOTL-I','HomOTL-II', 'Multitransfer1', 'Multitransfer2', 'Location', 'Northwest');
xlabel('Number of samples');
ylabel('average time cost (log_{10} t)')
grid

close(figure(gcf));

fprintf(1,'-------------------------------------------------------------------------------\n');
fprintf(1,'number of mistakes,            size of support vectors,           cpu running time\n');
fprintf(1,'PA             %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_PA1)/m*100,  std(err_PA1)/m*100, mean(nSV_PA1), std(nSV_PA1), mean(time_PA1), std(time_PA1));
fprintf(1,'PAIO             %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_PAI)/m*100,  std(err_PAI)/m*100, mean(nSV_PAI), std(nSV_PAI), mean(time_PAI), std(time_PAI));
fprintf(1,'HomOTL-I       %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_OTL)/m*100,  std(err_OTL)/m*100, mean(nSV_OTL), std(nSV_OTL), mean(time_OTL), std(time_OTL));
fprintf(1,'HomOTL-II      %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_OTL2)/m*100, std(err_OTL2)/m*100, mean(nSV_OTL2), std(nSV_OTL2), mean(time_OTL2), std(time_OTL2));
fprintf(1,'Multitransfer-I  %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_MT1)/m*100,   std(err_MT1)/m*100, mean(nSV_MT1), std(nSV_MT1), mean(time_MT1), std(time_MT1));
fprintf(1,'Multitransfer-II  %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f\n', mean(err_MT2)/m*100,   std(err_MT2)/m*100, mean(nSV_MT2), std(nSV_MT2), mean(time_MT2), std(time_MT2));
fprintf(1,'-------------------------------------------------------------------------------\n');
%}
