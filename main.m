% close;
% clear;
% clc;
func_num=15;
D=30;
Xmin=-100;
Xmax=100;
pop_size=100;
iter_max=500;
runs=25;
fhd=str2func('cec15_func');
fbest=zeros(func_num,runs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%IFMHDPSO
static_IFMHDPSO=zeros(func_num,2);
xbest_IFMHDPSO=cell(func_num,1);
curve_IFMHDPSO=zeros(func_num,iter_max);
for i=1:func_num
    func_id=i;
    xbest=zeros(runs,D);
    curve_temp=zeros(1,iter_max);
    parfor j=1:runs
        % gbest     Best Solution
        % gbestval  Best Fitness
        % curve     The optimal fitness for each iteration (used to draw the curve) 
        [gbest,gbestval,curve]= IFMHDPSO(fhd,func_id,D,Xmin,Xmax,iter_max,pop_size);
        xbest(j,:)=gbest;
        fbest(i,j)=gbestval;
        curve_temp=curve_temp+curve;
    end
    static_IFMHDPSO(i,:)=[mean(fbest(i,:)),sum((fbest(i,:)-mean(fbest(i,:))).^2)/runs];
    xbest_IFMHDPSO{i,1}=xbest;
    curve_IFMHDPSO(i,:)=curve_temp/runs;
end
if exist(['static/',num2str(30),'D/temp'],'dir')==0
    mkdir(['static/',num2str(30),'D/temp']);
end
if exist(['xbest/',num2str(30),'D/temp'],'dir')==0
    mkdir(['xbest/',num2str(30),'D/temp']);
end
if exist(['curve/',num2str(30),'D/temp'],'dir')==0
    mkdir(['curve/',num2str(30),'D/temp']);
end
save(['static/',num2str(D),'D/temp/static_IFMHDPSO.mat'],"static_IFMHDPSO");
save(['xbest/',num2str(D),'D/temp/xbest_IFMHDPSO.mat'],"xbest_IFMHDPSO");
save(['curve/',num2str(D),'D/temp/curve_IFMHDPSO.mat'],"curve_IFMHDPSO");