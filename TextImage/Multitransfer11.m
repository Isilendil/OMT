function [classifier, err_count, run_time, mistakes, mistakes_idx, SVs, TMs] = Multitransfer11(Y, K1, K2, P_it, text_gnd, options, id_list,classifier)
% HomOTL1: HomOTL-I
%--------------------------------------------------------------------------
% Input:
%        Y:    the vector of lables
%        K:    precomputed kernel for all the example, i.e., K_{ij}=K(x_i,x_j)
%  id_list:    a randomized ID list
%  options:    a struct containing rho, sigma, C, n_label and n_tick;
% Output:
%   err_count:  total number of training errors
%    run_time:  time consumed by this algorithm once
%    mistakes:  a vector of mistake rate
% mistake_idx:  a vector of number, in which every number corresponds to a
%               mistake rate in the vector above
%         SVs:  a vector records the number of support vectors
%     size_SV:  the size of final support set
%--------------------------------------------------------------------------

%% initialize parameters
eta = 1/2;
Number_old=options.Number_old;
C = options.C; % 1 by default
T_TICK = options.t_tick;
alpha1 = classifier.alpha;
SV1 = classifier.SV;
alpha2 = [];
SV2 = [];
ID = id_list;
err_count = 0;
mistakes = [];
mistakes_idx = [];
SVs = [];
TMs=[];
%=========================
w_1t=1/3;
w_2t=1/3;
w_3t=1/3;
K = options.K;
%=========================

t_tick = T_TICK; %10;

%% loop
tic
for t = 1:length(ID),
    id = ID(t);
    if (isempty(alpha1)), % init stage
        f1_t = 0;
    else
        k1_t = K1(id,SV1(:))';
        f1_t = alpha1*k1_t;            % decision function
    end
    
    id2=id-Number_old;
    if (isempty(alpha2)),
        f2_t=0;
    else
        k2_t=K2(id2,SV2(:))';
        f2_t=alpha2*k2_t;
    end
    
    hat_f1=max(0,min(1,(f1_t+1)/2));
    hat_f2=max(0,min(1,(f2_t+1)/2));
		%======================================
		f3_t = 0;
		similarity = P_it(id,:);
    for iter = 1 : K
			  min_value = 100000000;
				min_index = 1;
		    for jter = 1 : length(similarity)
					  if (similarity(jter) < min_value)
							  min_value = similarity(jter);
								min_index = jter;
						end
				end
				f3_t = f3_t + text_gnd(min_index);
				similarity(min_index) = 100000000000;
		end
		f3_t = f3_t / K;
		hat_f3 = max(0, min(1, (f3_t+1)/2));

    f_t=w_1t*hat_f1+w_2t*hat_f2+w_3t*hat_f3-1/2;
		%======================================
    
    hat_y_t = sign(f_t);        % prediction
    if (hat_y_t==0)
        hat_y_t=1;
    end
    % count accumulative mistakes
    if (hat_y_t~=Y(id)),
        err_count = err_count + 1;
    end
    
		%==================================
    ell_1=(hat_f1-(Y(id)+1)/2)^2;
    ell_2=(hat_f2-(Y(id)+1)/2)^2;
    ell_3=(hat_f3-(Y(id)+1)/2)^2;
    w_1t=w_1t*exp(-eta*ell_1);
    w_2t=w_2t*exp(-eta*ell_2);
    w_3t=w_3t*exp(-eta*ell_3);
    sum_w=w_1t+w_2t+w_3t;
    w_1t=w_1t/sum_w;
    w_2t=w_2t/sum_w;
    w_3t=w_3t/sum_w;
		%==================================
    
    l2_t = max(0,1-Y(id)*f2_t);   % hinge loss
    if (l2_t>0)
        % update
        s2_t=K2(id2,id2);
        gamma_t = min(C,l2_t/s2_t);
        alpha2 = [alpha2 Y(id)*gamma_t;];
        SV2 = [SV2 id2];
    end
    run_time=toc;
    
    if t<T_TICK
        if (t==t_tick)
            mistakes = [mistakes err_count/t];
            mistakes_idx = [mistakes_idx t];
            SVs = [SVs length(SV1)+length(SV2)];
            TMs=[TMs run_time];
            
            t_tick=2*t_tick;
            if t_tick>=T_TICK,
                t_tick = T_TICK;
            end
            
        end
    else
        if (mod(t,t_tick)==0)
            mistakes = [mistakes err_count/t];
            mistakes_idx = [mistakes_idx t];
            SVs = [SVs length(SV1)+length(SV2)];
            TMs=[TMs run_time];
        end
    end
    
    
end
classifier.SV1 = SV1;
classifier.SV2 = SV2;
classifier.alpha1 = alpha1;
classifier.alpha1 = alpha2;
fprintf(1,'The number of mistakes = %d\n', err_count);
run_time = toc;
