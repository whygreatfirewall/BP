clc
clear
close

%% ����Ԥ����
%ԭʼ���ݶ���
data_raw=xlsread('C:\Users\������\Desktop\������\����matlab��bp�㷨����\rawdata');

%���������Ŀ�������
p=data_raw(:,2:4)';  %������ת������Ϊmatlab�Դ��ĺ���Ҫ��ÿ�������������룬����excel�еİ�������
t=data_raw(:,5:6)';  %ͬ��

%����������Ŀ�������й�һ��
y_max=1;  y_min=-1;                    %��һ�����ֵ����Сֵ         
[pn,pPS]=mapminmax(p,y_min,y_max);  %�ú�����һ����ʽ��pn=2*(p-minp)/(maxp-minp)-1��ӳ������Ϊ[-1,1];y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin
[tn,tPS]=mapminmax(t,y_min,y_max); 

%Ϊ�����ӳ���Ŀ��޸��ԺͿɶ��ԣ����µĿɶ��Ըߵı���װ��ԭ���Ĺ�һ��������
SamNum=size(p,2);        %ѵ����������
TestSamNum=size(t,2);    %������������
InDim=size(p,1);         %���������������������ȣ���������ά�ȣ�
OutDim=size(t,1);        %����������lable�������ȣ��������ά�ȣ�
SamIn=pn;                %ѵ��������
SamOut=tn;               %ѵ�������
TestSamIn=pn;            %���Լ�����
TestSamOut=tn;           %���Լ����

% %���Ѿ���һ������������������Ϊ������ݵĲ��Լ���ѵ������ȫһ�����������������Ϊ�˷�ֹ�����
% rng('shuffle')    %���ϰ�matlab��rand('state',sum(100*clock))����ԭ����񣬸���ϵͳʱ�����Ӳ����������
% NoiseVar=0.01;    %����ǿ��Ϊ0.01����͹�һ�����ȶ�Ӧ��
% Noise=NoiseVar.*randn(size(SamOut));    %��������
% SamOut=SamOut+Noise;      %�������ӵ�ѵ����lable��
% TestSamOut=SamOut;        %���Լ���ѵ��������һ��

%% ��������
%��Ҫ���������
mynet=[InDim, 8 ,OutDim];  %����������Ҫ�����bp����Ĳ�����ÿ��Ľڵ�������matlab�Դ���newff���Ǹ�����һ��
net_len=length(mynet);     %��¼һ���������ĳ���

%��������ṹ����
net_W=cell(1,length(mynet)-1);   %Ȩ��W������ʼ��
net_B=cell(1,length(mynet)-1);   %ƫ��B������ʼ��
net_in=cell(1,length(mynet));  %ÿ������ڵ�������ʼ������
net_out=cell(1,length(mynet));  %ÿ������ڵ�������ʼ������

for i=1:(length(mynet)-1)        
    net_W(i)={0.5.*rand(mynet(i+1),mynet(i))-0.1};     %���������W����
    net_B(i)={0.5.*rand(mynet(i+1),1)-0.1};            %���������B����
end

%��������ѵ������
net_lr=0.35;           %ѧϰ��
net_MaxEpochs=50000;    %���ѵ������
net_E0=0.65*10^(-3);    %�����Ĺ�һ����Ĳв�ƽ����ѵ��Ŀ�ָ꣨���е���������Ĳв�ƽ������ӣ�����SSE

%����һ����������¼ÿ��ѵ����SSE
SSEHistory=[];  %�ȳ�ʼ����ÿ��ѵ��ѭ�����һ��ʱ��չ����

%% ��ʼѵ��
% net_in{1}=net_W{1}*SamIn+repmat(net_B{1},1,SamNum);   %�������һ���������г�ʼ��

for i=1:net_MaxEpochs
    %% ǰ�򴫵ݽ׶�
    %����㣨��һ�㣩������Ϣ���ȼ���purelin����
    net_in{1}=SamIn;       %��һ�㣨����㣩��������г�ʼ��
    net_out{1}=net_in{1};  %��һ�㣨����㣩�����ֱ�����������ȣ��ȼ���'purelin'�����
    %��������������������Ĵ���
    for j=2:(length(mynet)-1)
        net_in{j}=net_W{j-1}*net_out{j-1}+repmat(net_B{j-1},1,SamNum);  %�����ϲ�����Ƴ����������
        net_out{j}=logsig(net_in{j});   %����Ϊͼ��������������ڵ�ļ����ͳһ����logsig����
    end
    %����㹹��
    net_in{length(mynet)}=net_W{length(mynet)-1}*net_out{length(mynet)-1}+repmat(net_B{length(mynet)-1},1,SamNum);  %�����ϲ�����Ƴ����������
    net_out{length(mynet)}=net_in{length(mynet)};  %�������������ֱ�����������ȣ��ȼ���'purelin'�����
    
    %���㲢��¼�������ѵ���������Ĳв�
    Error=SamOut-net_out{end};     %ÿһ��ѵ���Ĳвÿ����ÿ�����������ѵ���еĲв
    SSE=sumsqr(Error);             %ÿһ��ѵ���Ĳв�ƽ���ͣ�ÿ����ÿ�����������ѵ���еĲв
    SSEHistory=[SSEHistory SSE];   %��ÿһ��ѵ���Ĳв�ƽ���ͱ�������������֮��ͼ֮��Ŀ��ӻ�����
    
    %% �ڵĸ��²���֮ǰ�����ж��Ƿ�ﵽ����Ҫ�����Ҫ�����ﵽ������ѭ��
    if SSE<net_E0,break,end
    
    %% ������׶�
    %����ÿ���Delta����
    net_Delta=cell(1,net_len);
    net_Delta{end}=Error;
    
    %�ѳ�������ÿ���Deltaֵȫ���õ�
    for j=net_len-1:-1:2
        net_Delta{j}=net_W{j}'*net_Delta{j+1}.*net_out{j}.*(1-net_out{j});  %�����������logsig�����󵼵Ľ��
    end
    
    %������Ľ���õ�dW��dB��ֵ
    net_dW=cell(1,length(mynet)-1);   %dW������ʼ��
    net_dB=cell(1,length(mynet)-1);   %dB������ʼ��
    for j=1:net_len-1
        net_dW{j}=net_Delta{j+1}*net_out{j}';      %����δ���ǹ���������ʧ����
        net_dB{j}=net_Delta{j+1}*ones(SamNum,1);   %����δ���ǹ���������ʧ����
    end
    
    %��������W��B����
    for j=1:net_len-1
    net_W{j}=net_W{j}+net_lr.*net_dW{j}./SamNum;            %����δ���ǹ���������ʧ����
    net_B{j}=net_B{j}+net_lr.*net_dB{j}./SamNum;            %����δ���ǹ���������ʧ����
    end
end

%% ����ѵ���������
    %����㣨��һ�㣩������Ϣ���ȼ���purelin����
    net_in{1}=TestSamIn;       %��һ�㣨����㣩��������г�ʼ��
    net_out{1}=net_in{1};       %��һ�㣨����㣩�����ֱ�����������ȣ��ȼ���'purelin'�����
    %��������������������Ĵ���
    for j=2:(length(mynet)-1)
        net_in{j}=net_W{j-1}*net_out{j-1}+repmat(net_B{j-1},1,SamNum);  %�����ϲ�����Ƴ����������
        net_out{j}=logsig(net_in{j});   %����Ϊͼ��������������ڵ�ļ����ͳһ����logsig����
    end
    %����㹹��
    net_in{length(mynet)}=net_W{length(mynet)-1}*net_out{length(mynet)-1}+repmat(net_B{length(mynet)-1},1,SamNum);  %�����ϲ�����Ƴ����������
    net_out{length(mynet)}=net_in{length(mynet)};  %�������������ֱ�����������ȣ��ȼ���'purelin'�����

%% ���չʾ
x=1990:2009;
a=mapminmax('reverse',net_out{end},tPS);  %�����һ������任��ԭ���ĳ߶�
newk=a(1,:);  %��һά���������������
newh=a(2,:);  %�ڶ�ά���������������
figure(2);
subplot(2,1,1);plot(x,newk,'r-o',x,t(1,:),'b--+');
legend('�������������','ʵ�ʿ�����')
xlabel('���');ylabel('������/����')
title('diy����ѵ����Ŀ������Ա�ͼ')
subplot(2,1,2);plot(x,newh,'r-o',x,t(2,:),'b--+');
legend('�������������','ʵ�ʻ�����')
xlabel('���');ylabel('������/����')
title('diy����ѵ����Ļ������Ա�ͼ')


