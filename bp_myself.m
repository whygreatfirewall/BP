clc
clear
close

%% 数据预处理
%原始数据读入
data_raw=xlsread('C:\Users\王宏晔\Desktop\教研室\基于matlab的bp算法轮子\rawdata');

%输入矩阵与目标矩阵构造
p=data_raw(:,2:4)';  %带矩阵转置是因为matlab自带的函数要求每个样本按列输入，而非excel中的按行输入
t=data_raw(:,5:6)';  %同上

%对输入矩阵和目标矩阵进行归一化
y_max=1;  y_min=-1;                    %归一化最大值与最小值         
[pn,pPS]=mapminmax(p,y_min,y_max);  %该函数归一化公式：pn=2*(p-minp)/(maxp-minp)-1，映射区间为[-1,1];y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin
[tn,tPS]=mapminmax(t,y_min,y_max); 

%为了增加程序的可修改性和可读性，用新的可读性高的变量装载原来的归一化的数据
SamNum=size(p,2);        %训练样本数量
TestSamNum=size(t,2);    %测试样本数量
InDim=size(p,1);         %单个样本的特征向量长度（网络输入维度）
OutDim=size(t,1);        %单个样本的lable向量长度（网络输出维度）
SamIn=pn;                %训练集输入
SamOut=tn;               %训练集输出
TestSamIn=pn;            %测试集输入
TestSamOut=tn;           %测试集输出

% %给已经归一化的输出添加噪声，因为这个数据的测试集和训练集完全一样，所以添加噪声是为了防止过拟合
% rng('shuffle')    %和老版matlab的rand('state',sum(100*clock))功能原理较像，根据系统时钟种子产生随机数。
% NoiseVar=0.01;    %噪声强度为0.01（这和归一化幅度对应）
% Noise=NoiseVar.*randn(size(SamOut));    %生成噪声
% SamOut=SamOut+Noise;      %噪声叠加到训练集lable上
% TestSamOut=SamOut;        %测试集与训练集保持一致

%% 构造网络
%想要构造的网络
mynet=[InDim, 8 ,OutDim];  %这里输入想要构造的bp网络的层数和每层的节点数，和matlab自带的newff的那个参数一样
net_len=length(mynet);     %记录一下这个网络的长度

%构造网络结构参数
net_W=cell(1,length(mynet)-1);   %权重W参数初始化
net_B=cell(1,length(mynet)-1);   %偏置B参数初始化
net_in=cell(1,length(mynet));  %每层网络节点的输入初始化（）
net_out=cell(1,length(mynet));  %每层网络节点的输出初始化（）

for i=1:(length(mynet)-1)        
    net_W(i)={0.5.*rand(mynet(i+1),mynet(i))-0.1};     %随机数构造W矩阵
    net_B(i)={0.5.*rand(mynet(i+1),1)-0.1};            %随机数构造B矩阵
end

%构造网络训练参数
net_lr=0.35;           %学习率
net_MaxEpochs=50000;    %最大训练次数
net_E0=0.65*10^(-3);    %样本的归一化后的残差平方和训练目标（指所有的样本输出的残差平方和相加），即SSE

%构造一个向量来记录每次训练的SSE
SSEHistory=[];  %先初始化，每次训练循环完成一次时扩展数据

%% 开始训练
% net_in{1}=net_W{1}*SamIn+repmat(net_B{1},1,SamNum);   %隐含层第一层的输入进行初始化

for i=1:net_MaxEpochs
    %% 前向传递阶段
    %输入层（第一层）接收信息，等价于purelin激活
    net_in{1}=SamIn;       %第一层（输入层）的输入进行初始化
    net_out{1}=net_in{1};  %第一层（输出层）的输出直接与输入层相等，等价于'purelin'激活函数
    %除了输出层以外的隐含层的传递
    for j=2:(length(mynet)-1)
        net_in{j}=net_W{j-1}*net_out{j-1}+repmat(net_B{j-1},1,SamNum);  %根据上层输出推出本层的输入
        net_out{j}=logsig(net_in{j});   %这里为图方便所有隐含层节点的激活函数统一用了logsig函数
    end
    %输出层构造
    net_in{length(mynet)}=net_W{length(mynet)-1}*net_out{length(mynet)-1}+repmat(net_B{length(mynet)-1},1,SamNum);  %根据上层输出推出本层的输入
    net_out{length(mynet)}=net_in{length(mynet)};  %这里输出层的输出直接与输入层相等，等价于'purelin'激活函数
    
    %计算并记录输出层与训练集样本的残差
    Error=SamOut-net_out{end};     %每一轮训练的残差（每列是每个样本在这次训练中的残差）
    SSE=sumsqr(Error);             %每一轮训练的残差平方和（每列是每个样本在这次训练中的残差）
    SSEHistory=[SSEHistory SSE];   %将每一轮训练的残差平方和保存下来，方便之后画图之类的可视化操作
    
    %% 在的更新参数之前首先判断是否达到了需要的误差要求，若达到则跳出循环
    if SSE<net_E0,break,end
    
    %% 反向传输阶段
    %构造每层的Delta矩阵
    net_Delta=cell(1,net_len);
    net_Delta{end}=Error;
    
    %把除输入层的每层的Delta值全部得到
    for j=net_len-1:-1:2
        net_Delta{j}=net_W{j}'*net_Delta{j+1}.*net_out{j}.*(1-net_out{j});  %后面的那项是logsig函数求导的结果
    end
    
    %由上面的结果得到dW与dB的值
    net_dW=cell(1,length(mynet)-1);   %dW参数初始化
    net_dB=cell(1,length(mynet)-1);   %dB参数初始化
    for j=1:net_len-1
        net_dW{j}=net_Delta{j+1}*net_out{j}';      %基于未考虑过拟合项的损失函数
        net_dB{j}=net_Delta{j+1}*ones(SamNum,1);   %基于未考虑过拟合项的损失函数
    end
    
    %更新网络W与B参数
    for j=1:net_len-1
    net_W{j}=net_W{j}+net_lr.*net_dW{j}./SamNum;            %基于未考虑过拟合项的损失函数
    net_B{j}=net_B{j}+net_lr.*net_dB{j}./SamNum;            %基于未考虑过拟合项的损失函数
    end
end

%% 测试训练后的网络
    %输入层（第一层）接收信息，等价于purelin激活
    net_in{1}=TestSamIn;       %第一层（输入层）的输入进行初始化
    net_out{1}=net_in{1};       %第一层（输出层）的输出直接与输入层相等，等价于'purelin'激活函数
    %除了输出层以外的隐含层的传递
    for j=2:(length(mynet)-1)
        net_in{j}=net_W{j-1}*net_out{j-1}+repmat(net_B{j-1},1,SamNum);  %根据上层输出推出本层的输入
        net_out{j}=logsig(net_in{j});   %这里为图方便所有隐含层节点的激活函数统一用了logsig函数
    end
    %输出层构造
    net_in{length(mynet)}=net_W{length(mynet)-1}*net_out{length(mynet)-1}+repmat(net_B{length(mynet)-1},1,SamNum);  %根据上层输出推出本层的输入
    net_out{length(mynet)}=net_in{length(mynet)};  %这里输出层的输出直接与输入层相等，等价于'purelin'激活函数

%% 结果展示
x=1990:2009;
a=mapminmax('reverse',net_out{end},tPS);  %将最后一层输出变换回原来的尺度
newk=a(1,:);  %第一维度输出，即客运量
newh=a(2,:);  %第二维度输出，即货运量
figure(2);
subplot(2,1,1);plot(x,newk,'r-o',x,t(1,:),'b--+');
legend('网络输出客运量','实际客运量')
xlabel('年份');ylabel('客运量/万人')
title('diy网络训练额的客运量对比图')
subplot(2,1,2);plot(x,newh,'r-o',x,t(2,:),'b--+');
legend('网络输出货运量','实际货运量')
xlabel('年份');ylabel('客运量/万人')
title('diy网络训练额的货运量对比图')


