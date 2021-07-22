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
    %r���Կ�������Ч�ʺ;��ȣ�Խ��r�ظ�������ͬ����ĸ���ԽС���������������ƣ����Ȼ��½���rԽС�ظ������ĸ���Խ�󣬵�������������������
         r=(ub-lb)/(2*10);
    end
    rng('default');
    rng('shuffle');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    explorerNum=ceil(pop*0.6);  % ̽�����Ա��
    minerNum=ceil(pop*0.4);  % �������Ա��
    epsilon=0.001;  % �������������ֵ����s���Ĳ�ֵ��С��epsilonʱ��Ϊ������ϣ�������ת��
    [explorer, eFit, miner, mBest_cur, mBestFit_cur, mBestPos_cur, mFit,v2,initFlag]=init(fhd, func_num, explorerNum, minerNum, dim, lb, ub, r);
    t=1;  % ��ǰ��������
    s=0;
    pNum=1;  % ��¼��ǰ��ֹ����forbiddenSet�н�ĸ���
    forbiddenSet=ones(maxIter,2*dim)*(max(ub)+1);  % ��ֹ����
    cur=zeros(1,maxIter);  % ��¼��������
    history=explorer;
    threshold=5;
    loB=floor(explorerNum/10)+1;  % ���Ž�Ⱥ����
    v=rand(explorerNum,dim).*r*2-r;
    [bestFitness,eBestPos]=min(eFit);
    bestX=explorer(eBestPos,:);
    history2=miner;
    corre=zeros(minerNum,2);
    checkedExplorer=mBest_cur;
    while t<=maxIter
        %��ʼ����̽����
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
        % ������ʷ����
        eFit=fhd(explorer',func_num)';
        eFit_history=fhd(history',func_num)';
        for i=1:explorerNum         
            if eFit(i,1)<eFit_history(i,:)
                history(i,:)=explorer(i,:);
            end
        end
        %------------------�������ż���--------------------%
        [sortExplorer_cur,indexE_cur]=sortrows([explorer,eFit],dim+1);
        % ��ǰ����������һ������sortExplorerǰdimά��������⣬���һά������Ӧ�ȣ�
        [sortExplorer,indexE]=sortrows([sortExplorer_last(1:loB,:);sortExplorer_cur(1:loB,:)],dim+1);
        temp_count=1;
        for ii=1:2*loB  % ����ǰloB����������
          flag=false;
          if indexE(ii)>loB
              temp_count=temp_count+1;
          else  % ֻ�и���֮ǰ�ļ���������Ҫ�滻����ǰ������
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
        %------------------����ȫ������------------------%
        if sortExplorer(1,dim+1)<bestFitness
          bestX=sortExplorer(1,1:dim);
          bestFitness=sortExplorer(1,dim+1);
        end
        %------------------���¿�����--------------------%
        mBestPos_last=mBestPos_cur;
        mBest_last=mBest_cur;
        mBestFit_last=mBestFit_cur;
        %------------------lamackѡ��--------------------%
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
            % ����Ҫ�ı��ֻ��==0��������ֱ��ͨ��postitive_eta�ۼӼ���
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
            else  % ���ݦǿ�ʼ��������
                [corre]=competition(minerNum,eta,1);
            end
        else  % ������Ӧ�ȿ�ʼ��������
            initFlag=1;
            eta=zeros(minerNum,1);
            tempFit=fhd(history2',func_num)';
            [corre]=competition(minerNum,tempFit,0);
        end
        %------------------����Ⱥ��ʼ---------------------%
        w2=rand(minerNum,1)*0.6+0.2;
        c1=1;
        c2=1;
        ran1=2*rand(minerNum,1);
        ran2=2*rand(minerNum,1);
        v2=w2.*v2+c1.*ran1.*(mBest_cur-miner)+c2.*ran2.*(history2(corre,:)-miner);
        v2=Bounds(v2,lb/10,ub/10);
        miner_temp=miner+v2;
        miner=Bounds(miner_temp,max([checkedExplorer-r;lb.*ones(1,dim)]),min([checkedExplorer+r;ub.*ones(1,dim)]));
        mFit_last=mFit;  % ������ʷ����
        %------------------������ʷ����--------------------%
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
        %------------------miner���¾ֲ�����--------------------%
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
        %------------------������ת���ж�--------------------%
        if abs(mBestFit_cur-mBestFit_last)<=epsilon
            s=s+1;
            if s>threshold %���������Ž�������ȴ���������ֵ��ת�Ƶ��µĿ��ɵ�
                %�ÿ�������õĽ��滻̽�������Ľ�
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
%% ��������
%--------------------------�߽纯��--------------------------
function [des]=Bounds(source,lb,ub)
    %�򵥵Ľ�Խ���ֵ���������ڱ߽��ϡ�
    flagUb=source>ub;
    flagLb=source<lb;
    des=(source.*(~(flagUb+flagLb)))+ub.*flagUb+lb.*flagLb;
end
%--------------------------��ʼ������--------------------------
function [explorer, eFit, miner, mBest, mBestFit, mBestPos_cur, mFit,v2, initFlag]=init(fhd, func_num, explorerNum, minerNum, dim, lb, ub, r)
    % exploer ̽����
    % eFit̽������Ӧ��
    % miner ������
    % mBest ���������Ž�
    % mBestFit ���������Ž����Ӧ��
    % mBestPos_cur ���������Ž��ڿ������е����
    % mFit ��������Ӧ��
    explorer=rand(explorerNum,dim).*(ub-lb)+lb;
    eFit=fhd(explorer',func_num)';
    [~,eBestPos]=min(eFit);
    eBest=explorer(eBestPos,:);
    %��ʼ���ɿ�����
    [miner, mBest, mBestFit, mBestPos_cur, mFit,v2,initFlag]=initMiner(fhd,func_num, eBest, minerNum, lb, ub, r,explorer);
end
%--------------------------��ʼ��������--------------------------
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
%--------------------------��ײ���--------------------------
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
%--------------------------ģ����--------------------------
function [checkedExplorer,finish]=checkFuzzy(explorer,bestFitness,worstFitness,r, forbiddenSet, pNum, fhd, func_num)
    finish=0;  % 0��ʾ���и��嶼�ڽ�ֹ������
    % miu��ʾ�����ȣ�lambda ��ʾ��������
    [explorerNum,dim]=size(explorer);
    eFit=fhd(explorer',func_num)';
    x1=log(eFit)/log(worstFitness);%��һ��
    miu1=exp(-x1.^2/0.18);%��Ӧ�ȵ�������
    lambda1=exp(-(x1-1).^2/0.18);%��Ӧ�ȵķ�������
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
    %�趨Ȩ��
    w=[0.8,0.1;0.8,0.1];
    %ȡ�Գ�Ȩϵ��omega=miu+lambda/2
    omega=w(:,1)+(1-sum(w,2));
    %Ȩ�ع�һ��
    omega=omega/sum(omega);
    %����ÿ���������ۺ�����ֵ
    evaluatedExplorer=[miu1*omega(1,1)+miu2*omega(2,1),lambda1*omega(1,1)+lambda2*omega(2,1)];
    %���������͸������
    idealSolution=[1,0];
    anti_idealSolution=[0,1];
    %�����ۺ�����ϵ��
    DAB=D(evaluatedExplorer,anti_idealSolution);
    xi=DAB./(DAB+D(evaluatedExplorer,idealSolution));
    %���������Ľӽ�������
    [~,index]=sort(xi,'descend');
    %ѡ���һ�����ڽ�ֹ�����еĸ���
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
%--------------------------ֱ��ģ�������뺯��---------------------
function [d]=D(A,B)
    [num,~]=size(A);
    d=-1*ones(num,1);
    pi1=1-sum(A,2);
    pi2=1-sum(B,2);
    d=(sum(abs(A-B),2)+abs(pi1-pi2))/2;
end
%--------------------------��������--------------------------
function [corre]=competition(minerNum,score,sign)
    % sign=0,��ʾС��ʤ��sign=1,��ʾ����ʤ
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