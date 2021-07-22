function [bestX, bestFitness, cur] = IFMHDPSO(fhd, func_num, dim, lb, ub, maxIter, pop, r)
%-------------------------------------------------------------------------%
%      Hybrid Double Particle Swarm Optimization Algorithm                % 
%    Based on Intuitionistic Fuzzy Memetic Framework(MHCHPSO)             %
%                                                                         %
%    Developed in MATLAB R2018a                                           %
%                                                                         %
%    Author and programmer: Kanqi Wang                                    %
%                                                                         %
%    e-Mail: wongkq@foxmail.com                                           %
%            wongkq@stumail.nwu.edu.cn                                    %
%                                                                         %
%    Programming dates: December 2018 to June 2019                        %
%                                                                         %
%-------------------------------------------------------------------------%    
%   Input:                                                                %
%       bestX represents the optimal solution                             %
%       bestFitnessrepresents the optimal fitness                         %
%       cur records the global optimal fitness of each iteration          %
%   Output:                                                               %
%       fhd represents the fitness function (CEC2015 is used by default ) %
%       func_num indicates the function number on CEC2015                 %
%       dim represents the dimension of the problem                       %
%       lb is a vector that represents the lower bound of the search      %
%          for each dimension of the problem                              %
%       ub is a vector that represents the upper bound of the search      %
%          in each dimension of the problem                               %
%       maxIter represents the maximum number of iterations               %
%       pop represents the total number of individuals, which will be     %
%           allocated to the explorer and the miner                       %
%           in a ratio of 6 to 4                                          %
%       r is a vector, which represents the optimization radius of the    % 
%         miner                                                           % 
%-------------------------------------------------------------------------%
    if ~exist('fhd', 'var')
        fhd=str2func('cec15_func');
        func_num=1;
        dim=30;
        lb=-100*ones(1,dim);
        ub=-lb;
    end
    if ~exist('maxIter','var')
        maxIter=500;
        pop=100;
    end
    if ~exist('r', 'var')
    %r可以控制搜索效率和精度，越大r重复搜索相同区域的概率越小，但由于跳出机制，精度会下降；r越小重复搜索的概率越大，但不会轻易跳出搜索。
         r=(ub-lb)/(2*10);
    end
    rng('default');
    rng('shuffle');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    explorerNum=ceil(pop*0.6);  % 探索组成员数
    minerNum=ceil(pop*0.4);  % 开发组成员数
    epsilon=0.001;  % 当开发组的最优值连续s代的差值都小于epsilon时认为开发完毕，开发组转移
    [explorer, eFit, miner, mBest_cur, mBestFit_cur, mBestPos_cur, mFit,v2,initFlag]=init(fhd, func_num, explorerNum, minerNum, dim, lb, ub, r);
    t=1;  % 当前迭代次数
    s=0;
    pNum=1;  % 记录当前禁止集合forbiddenSet中解的个数
    forbiddenSet=ones(maxIter,2*dim)*(max(ub)+1);  % 禁止集合
    cur=zeros(1,maxIter);  % 记录收敛曲线
    history=explorer;
    threshold=5;
    loB=floor(explorerNum/10)+1;  % 最优解群数量
    v=rand(explorerNum,dim).*r*2-r;
    [bestFitness,eBestPos]=min(eFit);
    bestX=explorer(eBestPos,:);
    history2=miner;
    corre=zeros(minerNum,2);
    checkedExplorer=mBest_cur;
    while t<=maxIter
        %开始跟新探索组
        [explorer,v,eFit]=checkCollision(explorer, loB, eFit, v, r, lb, ub, fhd, func_num);
        [sortExplorer_last,indexE_last]=sortrows([explorer,eFit],dim+1);
        bestGroup=sortExplorer_last(1:loB,1:end-1);           
        w=rand(explorerNum,1)*0.6+0.4;
        c1=2;
        c2=2;
        ran1=2*rand(explorerNum,1);
        ran2=2*rand(explorerNum,1);
        ran4=floor(rand(explorerNum,1).*loB)+1;
        v=w.*v+c1*ran1.*(history-explorer)+c2*ran2.*(bestGroup(ran4,:)-explorer);
        v=Bounds(v,lb/10,ub/10);
        explorer_temp=explorer+v;
        explorer=Bounds(explorer_temp,lb,ub);
        % 更新历史最优
        eFit=fhd(explorer',func_num)';
        eFit_history=fhd(history',func_num)';
        for i=1:explorerNum         
            if eFit(i,1)<eFit_history(i,:)
                history(i,:)=explorer(i,:);
            end
        end
        %------------------保留最优集团--------------------%
        [sortExplorer_cur,indexE_cur]=sortrows([explorer,eFit],dim+1);
        % 将前后两组混合在一起排序。sortExplorer前dim维分量保存解，最后一维保存适应度，
        [sortExplorer,indexE]=sortrows([sortExplorer_last(1:loB,:);sortExplorer_cur(1:loB,:)],dim+1);
        temp_count=1;
        for ii=1:2*loB  % 保留前loB各集团最优
          flag=false;
          if indexE(ii)>loB
              temp_count=temp_count+1;
          else  % 只有更新之前的集团最优需要替换到当前集团中
              for jj=1:ii-1
                  if indexE(jj)>loB
                      if indexE_cur(indexE(jj)-loB)==indexE_last(indexE(ii))
                          flag=true;
                          break;
                      end
                  end
              end
              if flag==false
                  explorer(indexE_last(indexE(ii)),:)=sortExplorer(ii,1:dim);
                  eFit(indexE_last(indexE(ii)))= fhd(explorer(indexE_last(indexE(ii)),:)',func_num)';
                  temp_count=temp_count+1;
              end
          end
          if temp_count>loB
              break;
          end
        end
        %------------------更新全局最优------------------%
        if sortExplorer(1,dim+1)<bestFitness
          bestX=sortExplorer(1,1:dim);
          bestFitness=sortExplorer(1,dim+1);
        end
        %------------------更新开发组--------------------%
        mBestPos_last=mBestPos_cur;
        mBest_last=mBest_cur;
        mBestFit_last=mBestFit_cur;
        %------------------lamack选择--------------------%
        if initFlag~=0
            temp_eta=-1*ones(minerNum,2);
            temp_eta(:,2)=0;
            for i=1:minerNum
              diff=mFit_last(i,1)-mFit(i,1);
              temp_eta(corre(i,1),1)=temp_eta(corre(i,1),1)+diff;
              temp_eta(corre(i,1),2)=temp_eta(corre(i,1),2)+1;
            end
            positiveNum=0;
            for i=1:minerNum
               if temp_eta(i,2)~=0
                   eta(i,1)=temp_eta(i,1)/temp_eta(i,2);
                   if eta(i,1)>0
                       positiveNum=positiveNum+1;
                       positive_eta(positiveNum,:)=[eta(i,1),i];
                   end
               elseif sum(i==setZero)==1
                   positiveNum=positiveNum+1;
                   positive_eta(positiveNum,:)=[0,i];
               else
                   eta(i,1)=-1;
               end
            end
            eta=eta.*(eta~=-1)+(eta==-1)*sum((eta~=-1).*eta)/sum(eta~=-1);
            % 由于要改变的只有==0的项所以直接通过postitive_eta累加即可
            if positiveNum>0
                positive_eta=positive_eta(1:positiveNum,:);
                if sum(positive_eta(:,1)>0)>4
                    positive_eta(:,1)=positive_eta(:,1)+(positive_eta(:,1)==0)*sum((positive_eta(:,1)>0).*positive_eta(:,1))/sum(positive_eta(:,1)>0);
                    for i=1:minerNum
                        temp_ran1=floor(rand(1)*positiveNum)+1;
                        temp_ran2=floor(rand(1)*positiveNum)+1;
                        if positive_eta(temp_ran1,1)>positive_eta(temp_ran2,1)
                            corre(i,1)=positive_eta(temp_ran1,2);
                        else
                            corre(i,1)=positive_eta(temp_ran2,2);
                        end
                    end
                else
                [corre]=competition(minerNum,eta,1);
                end
            else  % 根据η开始联赛机制
                [corre]=competition(minerNum,eta,1);
            end
        else  % 根据适应度开始联赛机制
            initFlag=1;
            eta=zeros(minerNum,1);
            tempFit=fhd(history2',func_num)';
            [corre]=competition(minerNum,tempFit,0);
        end
        %------------------粒子群开始---------------------%
        w2=rand(minerNum,1)*0.6+0.2;
        c1=1;
        c2=1;
        ran1=2*rand(minerNum,1);
        ran2=2*rand(minerNum,1);
        v2=w2.*v2+c1.*ran1.*(mBest_cur-miner)+c2.*ran2.*(history2(corre,:)-miner);
        v2=Bounds(v2,lb/10,ub/10);
        miner_temp=miner+v2;
        miner=Bounds(miner_temp,max([checkedExplorer-r;lb.*ones(1,dim)]),min([checkedExplorer+r;ub.*ones(1,dim)]));
        mFit_last=mFit;  % 保留历史最优
        %------------------更新历史最优--------------------%
        temp_count=0;
        temp_recode=ones(minerNum,1);
        mFit=fhd(miner',func_num)';
        mFit_history=fhd(history2',func_num)';
        for i=1:minerNum
            if mFit(i,1)<mFit_history(i,:)
                history2(i,:)=miner(i,:);
                temp_count=temp_count+1;
                temp_recode(temp_count,1)=i;
            end    
        end
        setZero=temp_recode(1:temp_count,1);
        %------------------miner更新局部最优--------------------%
        [mBestFit_cur,mBestPos_cur]=min(mFit);
        if mBestFit_last<mBestFit_cur
            mBestFit_cur=mBestFit_last;
            mBest_cur=mBest_last;
            mBestPos_cur=mBestPos_last;
            miner(mBestPos_last,:)=mBest_last;
        else
            mBest_cur=miner(mBestPos_cur,:) ;
        end
        if mBestFit_cur<bestFitness
            bestX=mBest_cur;
            bestFitness=mBestFit_cur;
        end
        %------------------开发组转移判断--------------------%
        if abs(mBestFit_cur-mBestFit_last)<=epsilon
            s=s+1;
            if s>threshold %开发的最优解连续相等次数超过阈值就转移到新的开采点
                %用开发组最好的解替换探索组最差的解
                [~,po]=max(eFit);
                explorer(po,:)=mBest_cur;
                eFit(po,:)=mBestFit_cur;
                worstFitness=max([eFit;mFit]);
                if mBestFit_cur>=checkedExplorer-epsilon
                    forbiddenSet(pNum,1:dim)=checkedExplorer;
                    forbiddenSet(pNum,dim+1:end)=r;
                    pNum=pNum+1;
                end
                [checkedExplorer,~]=checkFuzzy(explorer,bestFitness,worstFitness,r, forbiddenSet, pNum, fhd, func_num);
                [miner, mBest_cur,mBestFit_cur, mBestPos_cur, mFit,v2,initFlag]=initMiner(fhd,func_num,checkedExplorer, minerNum, lb, ub, r, explorer, v);
                s=0; 
            end
        else
            s=0;
        end
        cur(1,t)=bestFitness;
        t=t+1;
    end
end
%% 辅助函数
%--------------------------边界函数--------------------------
function [des]=Bounds(source,lb,ub)
    %简单的将越界的值重新设置在边界上。
    flagUb=source>ub;
    flagLb=source<lb;
    des=(source.*(~(flagUb+flagLb)))+ub.*flagUb+lb.*flagLb;
end
%--------------------------初始化函数--------------------------
function [explorer, eFit, miner, mBest, mBestFit, mBestPos_cur, mFit,v2, initFlag]=init(fhd, func_num, explorerNum, minerNum, dim, lb, ub, r)
    % exploer 探索组
    % eFit探索组适应度
    % miner 开发组
    % mBest 开发组最优解
    % mBestFit 开发组最优解的适应度
    % mBestPos_cur 开发组最优解在开发组中的序号
    % mFit 开发组适应度
    explorer=rand(explorerNum,dim).*(ub-lb)+lb;
    eFit=fhd(explorer',func_num)';
    [~,eBestPos]=min(eFit);
    eBest=explorer(eBestPos,:);
    %开始生成开采组
    [miner, mBest, mBestFit, mBestPos_cur, mFit,v2,initFlag]=initMiner(fhd,func_num, eBest, minerNum, lb, ub, r,explorer);
end
%--------------------------初始化开发组--------------------------
function [miner, mBest, mBestFit, mBestPos_cur, mFit, v2,initFlag]=initMiner(fhd,func_num, eBest, minerNum, lb, ub, r, explorer, v)
    count=1;    
    [explorerNum,dim]=size(explorer);
    miner=ones(minerNum,dim).*ub+1;
    v2=ones(minerNum,dim);
    if exist('explorer','var') && exist('v','var')
        for i=1:explorerNum
            if count>minerNum
                break;
            end
            if prod(abs(explorer(i,:)-eBest)<2*r,2)==1
                miner(count,:)=explorer(i,:);
                v2(count,:)=v(count,:);
                count=count+1;
            end
        end
    end
    miner(count:end,:)=eBest+(2*rand(minerNum-count+1,dim).*r-r);
    v2(count:end,:)=rand(minerNum-count+1,dim).*(ub-lb)/20+lb;
    miner=Bounds(miner,max([eBest-r;lb.*ones(1,dim)]),min([eBest+r;ub.*ones(1,dim)]));
    miner(end,:)=eBest;
    mFit=fhd(miner',func_num)';
    [mBestFit,mBestPos_cur]=min(mFit);
    mBest=miner(mBestPos_cur,:);
    initFlag=0;
end
%--------------------------碰撞检测--------------------------
function [explorer,v,eFit]=checkCollision(explorer, loB, eFit, v, r, lb, ub, fhd,func_num)
    [explorerNum,dim]=size(explorer);
    testX=explorer(randperm(explorerNum,1),:);
    count=sum(prod(abs(explorer-testX)<r,2)==1);
    if count<explorerNum*0.6
        return;
    end
    [~,index]=sortrows([explorer,eFit],dim+1);
    copyExplorer=explorer;
    copyV=v;
    m=zeros(explorerNum,1);
    m=(eFit+0.001).^-1/sum((eFit+0.001).^-1);
    for i=1:explorerNum
        if prod(i-index(1:loB))==0
            continue;
        end
        ui=zeros(1,dim);
        count=0;
        for j=1:explorerNum
            isCollision=prod(abs(copyExplorer(i,:)-copyExplorer(j,:))<=r,2);
            if isCollision==1 && i~=j
                ui=ui+(v(i,:)*(m(i,1)-m(j,1))+2*m(j,1)*copyV(j,:))/(m(i,1)+m(j,1));
                count=count+1;
            end
        end
        if count>0
            ui=Bounds(ui,lb/10,ub/10);
            v(i,:)=ui;
            explorer(i,:)=Bounds(copyExplorer(i,:)+ui, lb, ub);
        end
    end
    eFit=fhd(explorer',func_num)';
end
%--------------------------模糊化--------------------------
function [checkedExplorer,finish]=checkFuzzy(explorer,bestFitness,worstFitness,r, forbiddenSet, pNum, fhd, func_num)
    finish=0;  % 0表示所有个体都在禁止集合中
    % miu表示隶属度，lambda 表示非隶属度
    [explorerNum,dim]=size(explorer);
    eFit=fhd(explorer',func_num)';
    x1=log(eFit)/log(worstFitness);%归一化
    miu1=exp(-x1.^2/0.18);%适应度的隶属度
    lambda1=exp(-(x1-1).^2/0.18);%适应度的非隶属度
    x2=zeros(explorerNum,1);
    for i=1:explorerNum
        x2(i,1)=sum(sum(abs(explorer(i,:)-explorer)<r))/dim;
    end
    if max(x2)<explorerNum/2
        E=explorerNum/2;
    else
        E=explorerNum;
    end
    E=max(x2);
    sigma=E/2.06;
    miu2=exp(-(x2-E).^2/sigma^2);
    lambda2=1-exp(-(x2-E).^2/(sigma+1).^2); 
    %设定权重
    w=[0.8,0.1;0.8,0.1];
    %取对称权系数omega=miu+lambda/2
    omega=w(:,1)+(1-sum(w,2));
    %权重归一化
    omega=omega/sum(omega);
    %计算每个方案的综合评价值
    evaluatedExplorer=[miu1*omega(1,1)+miu2*omega(2,1),lambda1*omega(1,1)+lambda2*omega(2,1)];
    %设置理想解和负理想解
    idealSolution=[1,0];
    anti_idealSolution=[0,1];
    %计算综合评价系数
    DAB=D(evaluatedExplorer,anti_idealSolution);
    xi=DAB./(DAB+D(evaluatedExplorer,idealSolution));
    %按对理想解的接近度排序
    [~,index]=sort(xi,'descend');
    %选择第一个不在禁止集合中的个体
    checkedExplorer=explorer(index(1),:);
    forSet=forbiddenSet(1:pNum,1:dim);
    forSet_r=forbiddenSet(1:pNum,dim+1:end);
    for i=1:explorerNum
        if (sum(prod(abs(explorer(index(i),:)-forSet)<=forSet_r,2))==0)
            checkedExplorer=explorer(index(i),:);
            finish=1;
            break ;
        end
    end
end
%--------------------------直觉模糊集距离函数---------------------
function [d]=D(A,B)
    [num,~]=size(A);
    d=-1*ones(num,1);
    pi1=1-sum(A,2);
    pi2=1-sum(B,2);
    d=(sum(abs(A-B),2)+abs(pi1-pi2))/2;
end
%--------------------------联赛机制--------------------------
function [corre]=competition(minerNum,score,sign)
    % sign=0,表示小者胜；sign=1,表示大者胜
    corre=zeros(minerNum,1);
    num=size(score,1);
    for i=1:minerNum
       temp_ran1=floor(rand(1)*num)+1;
       temp_ran2=floor(rand(1)*num)+1;
       if (score(temp_ran1,1)<score(temp_ran2,1) && sign==0) || (score(temp_ran1,1)>score(temp_ran2,1) && sign==1)
           corre(i,1)=temp_ran1;
       else
           corre(i,1)=temp_ran2;
       end
    end
end