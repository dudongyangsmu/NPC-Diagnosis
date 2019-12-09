close all; clear; clc;

% add  fslib feast¡¢libsvm toolbox path
addpath(genpath('FSLib_v4.2'));
addpath(genpath('FEAST-v2.0.0_1'));
addpath(genpath('libsvm-3.22'));

dataStr='Fea_IBSI_tr_te_t5_CI';    
mkdir(['./results/',dataStr]);
mkdir(['./results/',dataStr,'/','Dtree/']);mkdir(['./results/',dataStr,'/','KNN/']);mkdir(['./results/',dataStr,'/','LDA/']);
mkdir(['./results/',dataStr,'/','LR/']);mkdir(['./results/',dataStr,'/','NBayes/']);mkdir(['./results/',dataStr,'/','Rforest/']);
mkdir(['./results/',dataStr,'/','SVM/']);


for k=5                           %  select the top kth non-redundant feature
    for p=1:6                    %  select the selection method
        %for iter=1:30
            for q=1:7            %  select the classifier
                tic
         
                    % load data
                     load('trainData.mat')
                     load('testData.mat')
 
                    % select the FS method
                    listFS = {'fisher','relieff','mrmr','mim','cmim','jmi'};
                    selection_method = listFS{p}; % Selected
                    
                    % select the classification method
                    listCM = {'LR','SVM','KNN','LDA','NBayes','Dtree','Rforest'};
                    methodID_cm =q;
                    classification_method = listCM{methodID_cm}; % Selected
                    
                    % feature ranking
                    fspara=10;
                    numToSelect=size(X_train,2);
                    if p<3
                        rank=featureSelect(X_train,Y_train,selection_method,fspara );
                    else
                        % feast requires that both the features and labels are integers
                        X_train_discre=uniformQuantization(X_train,5);                           
                        rank=feast(selection_method,numToSelect,X_train_discre,Y_train);
                    end
                    
                    %corr(X_train(:,rank(1:k)))
                    
                    % remove highly related features
                    X_train=X_train(:,rank);
                    [spear,pval] = corr(X_train,'type','Spearman');
                    spear_tri = tril(spear);
                    n = size(spear,1);                                             %Feas number
                    abandon = [];                                                     %abandon feature ID
                    num=1;
                    for i=2:n
                        spear_tem=spear_tri(i,:);
                        for j=1:(i-1)
                            if abs(spear_tem(j))>0.7
                                abandon(num) = i;
                                num=num+1;
                            end
                        end
                    end
                    abandon=unique(abandon);
                    Fea_idx = setdiff((1:n), abandon) ;
                    X_train=X_train(:,Fea_idx);
                    X_test=X_test(:,rank(Fea_idx));
                    
                    %corr(X_train(:,1:k))
                    
                    %classifier requires that label is 1 or 2
                    Y_train_C=Y_train;
                    Y_train_C(Y_train==1)=2;
                    Y_train_C(Y_train==-1)=1;
                    
                    Y_test_C=Y_test;
                    Y_test_C(Y_test==1)=2;
                    Y_test_C(Y_test==-1)=1;
                    
                    % classification methods
                    switch methodID_cm
                        case 1
                            % LR
                            B=mnrfit(X_train(:,1:k),Y_train_C);
                            prob=mnrval(B,X_test(:,1:k));
                            prob_train=mnrval(B,X_train(:,1:k));
                        case 2
                            % SVM-RBF
                            model = libsvmtrain(Y_train_C,X_train(:,1:k),'-b  1 ');
                            [label,accuracy,prob] =svmpredict(Y_test_C,X_test(:,1:k),model,'-b 1');
                            [label_train,accuracy_train,prob_train] =svmpredict(Y_train,X_train(:,1:k),model,'-b 1');
                        case 3
                            % KNN
                            model=fitcknn(X_train(:,1:k),Y_train_C,'NumNeighbors',3);
                            [label,prob,cost] = predict(model,X_test(:,1:k));
                            [label_train,prob_train,cost_train] = predict(model,X_train(:,1:k));
                        case 4
                            % LDA
                            model=fitcdiscr(X_train(:,1:k),Y_train_C);
                            [label,prob,cost] = predict(model,X_test(:,1:k));
                            [label_train,prob_train,cost_train] = predict(model,X_train(:,1:k));
                        case 5
                            % NB
                            model=fitcnb(X_train(:,1:k),Y_train_C);
                            [label,prob,cost] = predict(model,X_test(:,1:k));
                            [label_train,prob_train,cost_train] = predict(model,X_train(:,1:k));
                        case 6
                            % DT
                            model=fitctree(X_train(:,1:k),Y_train_C);
                            [label,prob,cost] = predict(model,X_test(:,1:k));
                            [label_train,prob_train,cost_train] = predict(model,X_train(:,1:k));
                        case 7
                            % RF  note: repeat 30 times to compute average/max
                            % AUC and test error values because of the
                            % randomness of random forest. Corresponding
                            % code needs to be tuned slightly.
                            model=TreeBagger(100,X_train(:,1:k),Y_train_C);
                            [label,prob] =  predict(model,X_test(:,1:k));
                            [label_train,prob_train] =  predict(model,X_train(:,1:k));
                    end
                    
                    % result of validation set
                    [X,Y,T,auc,OPTROCPT] = perfcurve_DDY(Y_test, prob(:,2),1,'NBoot',100,'BootType','bca');  
                    [acc,~,T_acc,~] = perfcurve_DDY(Y_test, prob(:,2),1,'XCrit','accu','NBoot',100,'BootType','bca');
                    
                    ind=(X(:,1)==OPTROCPT(1))&(Y(:,1)==OPTROCPT(2));
                    spe=1-X(ind,:);
                    sen=Y(ind,:);
                    ACC=acc(ind,:);
                    
                    % result of training set
                    [XX_train,YY_train,TT_train,auc_train,OPTROCPT_train] = perfcurve_DDY(Y_train, prob_train(:,2),1,'NBoot',100,'BootType','bca');%,'NBoot',1000
                    [acc_train,~,T_acc_train,~] = perfcurve_DDY(Y_train, prob_train(:,2),1,'XCrit','accu','NBoot',100,'BootType','bca');
                    
                    ind_train=(XX_train(:,1)==OPTROCPT_train(1))&(YY_train(:,1)==OPTROCPT_train(2));
                    spe_train=1-XX_train(ind_train,:);
                    sen_train=YY_train(ind_train,:);
                    ACC_train=acc_train(ind_train,:);
                    
                    % save result
                    stats.selection_method=selection_method;
                    stats.classification_method=classification_method;
                    stats.numFeatures=k;
                    stats.featureIndex=rank(Fea_idx(1:k));
                    
                    stats.auc=auc;
                    stats.spe=spe;
                    stats.sen=sen;
                    stats.ACC= ACC;
                    stats.prob=prob;
                    
                    stats.optimalpoint=OPTROCPT;
                    stats.T=T;
                    stats.X=X;
                    stats.Y=Y;
                    
                    stats.auc_train=auc_train;
                    stats.spe_train=spe_train;
                    stats.sen_train=sen_train;
                    stats.acc_train=ACC_train;
                    stats.prob_train=prob_train;
                    
                    stats.optimalpoint_train=OPTROCPT_train;
                    stats.T_train=TT_train;
                    stats.X_train=XX_train;
                    stats.Y_train=YY_train;
             
                toc
                fileName=['./results/',dataStr,'/',classification_method,'/',selection_method,classification_method];
                saveStats(  fileName,stats);
            end
        %end
    end
end







