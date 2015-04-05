function [u_rank, v_rank]= CTLearn(min_val, P_11, P_22, P_12, P_21,  u, v, u_0, v_0 , alpha, beta, gamma )
%    Syntax
%
%        [u_rank, v_rank]= CTLearn(min_val, P_11, P_22, P_12, P_21,  u, v, u_0, v_0 , alpha, beta, gamma )
%
%    Description
%
%       inputs,
%           min_val  - the minimal change of the successive computed probabilities U(t) - U(t-1), where U is the state probabilites
%           P_11     - A M1xM1 matrix, M1 is the number of instances of 1st task, Q is the possible class number, P_11 is the transition probability matrix of 1st task
%           P_22     - A M2xM2 matrix, M2 is the number of instances of 2nd task, P_22 is the transition probability matrix of 2nd task
%           P_12     - A M1xM2 matrix, the transition probability matrix of 1st task to 2nd task
%           P_21      - A M2xM1 matrix, the transition probability matrix of 2nd task to 1st task
%           u         - A M1xQ matrix, the initial state probability of 1st task 
%           v         - A M2xQ matrix, the initial state probability of 2nd task
%           u_0       - A M1xQ matrix, the class label distribution matrix of 1st task
%           v_0       - TA M2xQ matrix, the class label distribution matrix of 1st task
%           alpha     - The restart parameter 0<alpha<1
%           beta      - The parameter to control the amount of knowledge to be transferred from 1nd task to 2nd task
%           gamma     - The parameter to control the amount of knowledge to be transferred from 2nd task to 1st task
%           classifier
%      and returns,
%           u_rank      - A MxQ matrix, the output of 1st learning task, the ith testing instance on the jth class is stored in u_rank(i,j)
%           v_rank      - A MxQ matrix, the output of 2nd learning task, the ith testing instance on the jth class is stored in v_rank(i,j)
    
diff = min_val;
class_num = size(u, 2);

for i=1:class_num         % for u
    sum_val = sum(u(:,i));
    if sum_val>0
        u(:,i) = u(:,i)/sum_val;
    end
end
for i=1:class_num         % for v
    sum_val = sum(v(:,i));
    if sum_val>0
        v(:,i) = v(:,i)/sum_val;
    end
end
for i=1:class_num         % for u_0
    sum_val = sum(u_0(:,i));
    if sum_val>0
        u_0(:,i) = u_0(:,i)/sum_val;
    end
end
for i=1:class_num         % for v_0
    sum_val = sum(v_0(:,i));
    if sum_val>0
        v_0(:,i) = v_0(:,i)/sum_val;
    end
end

u_length = size(u, 1);
v_length = size(v, 1);

X = [u; v];
P = [u_0; v_0];
% iterative
iter_num = 0;

while diff>=min_val
    if iter_num > 100
        break;
    end
    
    X_2 = (1-alpha) * [ beta*P_11, (1-beta)*P_12; (1-gamma)*P_21, gamma*P_22 ]*X + alpha*P;
    
    % normalize
    for i=1:class_num
        sum_u = sum(X_2(1:u_length,i));
        sum_v = sum(X_2(u_length+1:u_length+v_length,i));
        if sum_u >0
            X_2(1:u_length,i) = X_2(1:u_length,i)/sum_u;
        end
        if sum_v >0
            X_2(u_length+1:u_length+v_length,i) = X_2(u_length+1:u_length+v_length,i)/sum_v;
        end
    end
    X_diff = abs(X - X_2);
    diff = sum(X_diff(:));
    X = X_2;
    iter_num = iter_num + 1;
    fprintf(1,'%s %d %s %d\n','iter_num:,',iter_num,',diff:,',diff)
    
end
u_rank = X_2(1:u_length,:);
v_rank = X_2(u_length+1:u_length+v_length,:);

end


