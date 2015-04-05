%**************************************************************************
% An example for the Co-Transfer Learning

% The code has been used in:
%  Qingyao Wu, Michael Ng, and Yunming Ye, "Co-Transfer Learning Using Coupled Markov Chains with Restart", IEEE Intelligent Systems, DOI: 10.1109/MIS.2013.32, 2013
%  Michael Ng, Qingyao Wu and Yunming Ye. "Co-Transfer Learning via Joint Transition Probability Graph Based Method". In: SIGKDD Workshop on CDKD'12, pp.1-9, 2012 
%
% ATTN:   You can feel free to use the package (for academic purpose only) at your own risk. 
% An acknowledge or citation to the above paper is required. 
% For other purposes, please contact Dr. Qingyao Wu (wuqingyao.china@gmail.com).

%       data set description,
%           text_fea  - A M1xN1 matrix, the ith instance is stored in text_fea{i,:), M1 is the number of instances of text task, N1 is the number of text features
%           text_gnd  - A M1x1 matrix, if the ith text instance belongs to the jth class, then text_gnd(i) equals j
%           image_fea - A M2xN2 matrix, the ith instance is stored in image_fea{i,:), M2 is the number of instances of image task, N2 is the number of image features
%           image_gnd   - A M2x1 matrix, if the ith image instance belongs to the jth class, then image_gnd(i) equals j

clc; clear;

% default parameter
alpha = 0.9;
beta = 0.1;
gama = 0.9;
min_val = 1E-15;
% flowers-tree data set
data_name = 'flowers-tree';
class_num = 2;

% data filename & affinity filename
data_file = ['data/' data_name , '.mat'] ;
affinity_file = ['affinity/' data_name , '_affinity.mat'] ;
split_file = ['split/' data_name , '.mat'] ;
% load data
c_data = load(data_file);
c_text_fea = c_data.text_fea;
c_text_gnd = c_data.text_gnd;
c_image_fea = c_data.image_fea;
c_image_gnd = c_data.image_gnd;
% calculate affinity matrix
if ~exist(affinity_file,'file')
    if ~exist('affinity', 'dir')
        mkdir('affinity')
    end
    calculate_affinity( data_file, affinity_file );
end
% load affinity
affinity_matrix = load(affinity_file);
% load similarity
P_ii = affinity_matrix.P_ii;
P_tt = affinity_matrix.P_tt;
P_ti = affinity_matrix.P_ti;
P_it = P_ti';
% normalize transition probability matrixs
for i=1:size(P_ii,2)
    sum_col = sum(P_ii(:,i));
    if sum_col~=0
        P_ii(:,i) = P_ii(:,i)/sum_col;
    end
end
for i=1:size(P_tt,2)
    sum_col = sum(P_tt(:,i));
    if sum_col~=0
        P_tt(:,i) = P_tt(:,i)/sum_col;
    end
end
for i=1:size(P_ti,2)
    sum_col = sum(P_ti(:,i));
    if sum_col~=0
        P_ti(:,i) = P_ti(:,i)/sum_col;
    end
end
for i=1:size(P_it,2)
    sum_col = sum(P_it(:,i));
    if sum_col~=0
        P_it(:,i) = P_it(:,i)/sum_col;
    end
end


rst_str_sum = '';
for proportion = 1:6  % varying number of training examples
    % instance number
    image_num = size(c_image_fea, 1);
    text_num = size(c_text_fea, 1);
    % construct q2, where q2 is the assigned vector of the 2nd class label
    c_image_q2 = zeros(image_num,1);
    c_text_q2 = zeros(text_num,1);
    if max(c_image_gnd)>1
        c_image_gnd = c_image_gnd -1;
    end
    for i=1:image_num
        if c_image_gnd(i) == 1
            c_image_q2(i) = 0;
        else
            c_image_q2(i) = 1;
        end
    end
    for i=1:text_num
        if c_text_gnd(i) == 1
            c_text_q2(i) = 0;
        else
            c_text_q2(i) = 1;
        end
    end
    
    % cross validation
    trails = 10;
    image_acc = zeros(trails,1);
    % load split
    c_splits = load(split_file);
    image_train_idx = c_splits.image_train_idx;
    image_test_idx = c_splits.image_test_idx;
    
    image_train_split = image_train_idx{proportion};
    image_test_split = image_test_idx{proportion};
    % train_num
    train_num = size(image_train_split,2);
    
    for t = 1:trails
        % training set
        image_train_ind = image_train_split(t,:);
        image_test_ind = image_test_split(t,:);
        text_train_ind = [];
        text_test_ind = 1:text_num;
        
        % construct the distribution matrix of the class labels
        u_0 = [c_image_q2, c_image_gnd];
        u_0(image_test_ind,:)=0;
        u = u_0;
        u(image_test_ind,:)=1/class_num;
        
        v_0 = [c_text_q2, c_text_gnd];
        v_0(text_test_ind,:)=0;
        v = v_0;
        v(text_test_ind,:)=1/class_num;
        
        
        % run MT-Learn
        tStart=tic;
        [u_rank,v_rank] = CTLearn(min_val, P_ii, P_tt, P_it, P_ti,  u, v, u_0, v_0, alpha, beta, gama);
        %         [u_rank,v_rank] = mav.solveMarkov2(P_ii, P_tt, P_it, P_ti,  u, v, u_0, v_0, alpha, beta, gama);
        
        time = toc(tStart);
        
        % Evaluation, compare true target label and output predict resutls
        image_target = c_image_gnd(image_test_ind,:);
        
        image_pred = u_rank(image_test_ind,:);
        image_output = zeros(size(image_pred,1), 1);
        % normalize by row
        for k=1:size(image_pred,1)
            sum_col = sum(image_pred(k,:));
            image_pred(k,:) = image_pred(k,:)/sum_col;
            if image_pred(k,1)>image_pred(k,2)  % first column corresponding to class 1
                image_output(k) = 0;
            else
                image_output(k) = 1;
            end
        end
        
        % calculate accuracy
        eva = ClassificationEva();
        image_acc(t) = 1 - eva.calTestError(image_target, image_output);
        %
    end
    
    m_image_acc = mean(image_acc);
    m_image_acc_std = std(image_acc);
    
    % output rst
    rst_str = [  data_name , ','  ,  num2str(train_num)  , ',' , num2str(m_image_acc) ]
    rst_str_sum = sprintf('%s\n%s',rst_str_sum, rst_str);
end
disp([ 'data, train_num, image_accuracy: ']);
disp(rst_str_sum);
clear;




