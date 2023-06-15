clear; close all; clc
randn('seed',0);
filter = singletargetFilter;
filter = filter.gen_model;

MCRuns = 1;     %蒙特卡洛运行次数

% 初始化
RMSE_posCKF1 = zeros(MCRuns,filter.K); % 滤波器1的位置均方根误差
RMSE_velCKF1 = zeros(MCRuns,filter.K); % 滤波器1的速度均方根误差
RMSE_posCKF2 = zeros(MCRuns,filter.K); % 滤波器2的位置均方根误差
RMSE_velCKF2 = zeros(MCRuns,filter.K); % 滤波器2的速度均方根误差
AVE_posRMSE1=0; 
AVE_velRMSE1=0; 
AVE_posRMSE2=0; 
AVE_velRMSE2=0;
for iMCruns = 1:MCRuns

    stateUpd_CKF1 = [ 1.5; 1.0; 0.21; 0.64];                 % 滤波器1的状态初始值   [ 1.5; 1.0; 0.21; 0.64]
    covarUpd_CKF1 = blkdiag(1.2*eye(3),0.5);                 % 滤波器1的状态方差初始值  blkdiag(10*eye(4),pi/90); (10,10,100,10,0.1)
    stateUpd_CKF2 = [ 1.6; 1.0; 0.11; 0.64];                 % 滤波器2的状态初始值   [ 0; 6; 0; 1; 0.02 ];
    covarUpd_CKF2 = blkdiag(1*eye(3),pi/90);                 % 滤波器2的状态方差初始值  blkdiag(10*eye(4),pi/90); (10,10,100,10,0.1)
    est_CKF11 = zeros(filter.targetStateDim,filter.K);       % 第1个滤波器的第1级状态估计存储向量
    est_CKF12 = zeros(filter.targetStateDim,filter.K);       % 第1个滤波器的第2级状态估计存储向量
    est_CKF21 = zeros(filter.targetStateDim,filter.K);       % 第2个滤波器的第1级状态估计存储向量
    est_CKF22 = zeros(filter.targetStateDim,filter.K);       % 第2个滤波器的第2级状态估计存储向量
    xpre1 = zeros(filter.targetStateDim,filter.K);           % 第1个滤波器的第1级滤波器一步预测状态估计值存储
    xpre2 = zeros(filter.targetStateDim,filter.K);           % 第2个滤波器的第1级滤波器一步预测状态估计值存储
    est_CKF_final = zeros(filter.targetStateDim,filter.K);   % 滤波器经过BP融合后的状态估计存储向量

    tCKF = 0;
    est_CKF11(:,1) = filter.truth_X(:,1);
    est_CKF12(:,1) = filter.truth_X(:,1);
    for k = 2:filter.K
       %%
        tic
        %================================ 第一级滤波器 %=========================
        % 第一个传感器两步状态预测
        [x1, p1] = filter.State_two_steps_prediction(stateUpd_CKF1, covarUpd_CKF1);
        
        % 第一个传感器一步状态预测
        zk = filter.meas1(:,k-1);
        if (k==2)
            zk_pre_last = filter.meas1(:,1);                     % 以真实量测作为初始时刻的量测预测 需要在后续进行更新  [0.2155; 6.7194]
            xpre1(:,k-1) = filter.truth_X(:,1);                  % 以真实状态作为初始时刻的状态预测 需要进行更新[5.9; 5.9; 1.0; 0.12; 0.02]
            p2 = blkdiag(0.25, 0.25 ,0.05 ,0.01);                 % 以两步状态预测方差作为上一步状态预测方差 blkdiag(10*eye(5))  p1
            pzz_last = blkdiag(1.6,3.6);                         % 10,pi/90
            pxv = filter.Q*(filter.S2);                          % 与量测更新时的计算方式一样
            N2 = [p2, pxv; pxv', filter.sigma_measNoise1];
        end
        xv = [xpre1(:,k-1); 0; 0 ];
        [x2,p2] = filter.State_one_steps_prediction(x1, p1, zk, zk_pre_last, pzz_last, xv, N2);
        xpre1(:,k) = x2;
        
        % 第一个传感器量测更新
        measurement = filter.meas1(:,k);                        % 结合当前时刻的量测数据进行更新
        Rv1 = filter.sigma_measNoise1;                          % 传感器1的量测噪声方差
        [x3,p3,K_gain,zpre, N3,Pzz,Pxz] = filter.measurement_update(x2, p2, measurement, Rv1);
        zk_pre_last = zpre;
        N2 = N3;
        pzz_last = Pzz;
        est_CKF11(:,k) = x3;
%         stateUpd_CKF1 = x3;
%         covarUpd_CKF1 = p3;
        
        %================================ 第二级滤波器 %=========================
        % 第一个传感器在线性空间下的一步状态预测
        if (k==2)
            x_final = filter.truth_X(:,1);      % x3
            p_final = blkdiag(0.55, 0.04 ,0.04 ,0.09);      % blkdiag(0.1*eye(5))    1.05, 0.40 ,0.54 ,0.09
        end
        [x4, p4] = filter.LS_prediction(x_final, p_final);
        
        % 第一个传感器在线性空间下的量测更新
%         test = [p4, filter.Q*filter.S2; (filter.Q*filter.S2)', filter.sigma_measNoise1];
        [x_final, p_final] = filter.LS_update(x4, p4, x3, K_gain, Rv1);
        stateUpd_CKF1 = x_final;     % 二级滤波后的状态迭代更新
        covarUpd_CKF1 = p_final;     % 二级滤波后的方差迭代更新
        est_CKF12(:,k) = x_final;    % 二级滤波后状态的存储
        
        %xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 第二个传感器开始 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        %================================ 第一级滤波器 %=========================
        % 第二个传感器两步状态预测
        [x21, p21] = filter.State_two_steps_prediction(stateUpd_CKF2, covarUpd_CKF2);
        
        % 第一个传感器一步状态预测
        zk2 = filter.meas2(:,k-1);
        if (k==2)
            zk_pre_last2 = filter.meas2(:,1);               % 以真实量测作为初始时刻的量测预测 需要在后续进行更新 [0.15; 7.78]
            xpre2(:,k-1) = filter.truth_X(:,1);             % 以真实状态作为初始时刻的状态预测 需要进行更新 [5.9; 5.9; 1.0; 0.12; 0.02]
            p22 = blkdiag(1.05, 0.55 ,1.04 ,1.1);    % 以两步状态预测方差作为上一步状态预测方差  p21 0.001*eye(5)
            pzz_last2 = blkdiag(1.8,3.6);                      % blkdiag(1,pi/90)
            pxv2 = filter.Q*(filter.S2);                    % 与量测更新时的计算方式一样
            N22 = [p22, pxv2; pxv2', filter.sigma_measNoise1];
        end
        xv2 = [xpre2(:,k-1); 0; 0 ];
          
        [x22,p22] = filter.State_one_steps_prediction(x21, p21, zk2, zk_pre_last2, pzz_last2, xv2, N22);
        xpre2(:,k) = x22;
        
        % 第一个传感器量测更新
        measurement2 = filter.meas2(:,k);  % 结合当前时刻的量测数据进行更新
        Rv2 = filter.sigma_measNoise2;     % 传感器2的量测噪声方差
        [x23,p23,K_gain2,zpre2, N23,Pzz2,Pxz2] = filter.measurement_update(x22, p22, measurement2, Rv2);
        zk_pre_last2 = zpre2;
        N22 = N23;
        pzz_last2 = Pzz2;
        est_CKF21(:,k) = x23;
%         stateUpd_CKF2 = x23;     % 二级滤波后的状态迭代更新
%         covarUpd_CKF2 = p23;     % 二级滤波后的方差迭代更新
        
        %================================ 第二级滤波器 %=========================
        % 第一个传感器在线性空间下的一步状态预测
        if (k==2)
            x_final2 = filter.truth_X(:,1);                     % x23
            p_final2 = blkdiag(1.05, 0.29 ,0.24 ,0.49);         % p23
        end
        [x24, p24] = filter.LS_prediction(x_final2, p_final2);
        
        % 第一个传感器在线性空间下的量测更新
        [x_final2, p_final2] = filter.LS_update(x24, p24, x23, K_gain2, Rv2);
        stateUpd_CKF2 = x_final2;     % 二级滤波后的状态迭代更新
        covarUpd_CKF2 = p_final2;     % 二级滤波后的方差迭代更新
        est_CKF22(:,k) = x_final2;    % 二级滤波后状态的存储
        
        
        % 滤波器1状态RMSE
        RMSE_posCKF1(iMCruns,k) = sqrt(sum((stateUpd_CKF1([1 3])-filter.truth_X([1 3],k)).^2));
        RMSE_velCKF1(iMCruns,k) = sqrt(sum((stateUpd_CKF1([2 4])-filter.truth_X([2 4],k)).^2));
        % 滤波器2状态RMSE
        RMSE_posCKF2(iMCruns,k) = sqrt(sum((stateUpd_CKF2([1 3])-filter.truth_X([1 3],k)).^2));
        RMSE_velCKF2(iMCruns,k) = sqrt(sum((stateUpd_CKF2([2 4])-filter.truth_X([2 4],k)).^2));
        
        tCKF = tCKF+toc;
        fprintf('K=%f\n',k);
        %%
    end
    
    %    disp(['Current Iteration is ',num2str(iMCruns),'.']);
    %
    disp('========================');
    disp('耗费时间/s：');
    disp(['CKF:',num2str(tCKF)]);
end
% delete(h);

RMSE_posCKF1 = mean(RMSE_posCKF1,1); RMSE_velCKF1 = mean(RMSE_velCKF1,1);     % 获取的每一列的均值
RMSE_posCKF2 = mean(RMSE_posCKF2,1); RMSE_velCKF2 = mean(RMSE_velCKF2,1);


for i = 1:filter.K
    
    AVE_posRMSE1 = AVE_posRMSE1 + RMSE_posCKF1(1,i); 
    AVE_velRMSE1 = AVE_velRMSE1 + RMSE_velCKF1(1,i); 
    AVE_posRMSE2 = AVE_posRMSE2 + RMSE_posCKF2(1,i); 
    AVE_velRMSE2 = AVE_velRMSE2 + RMSE_velCKF2(1,i);
end
AVE_posRMSE1=AVE_posRMSE1/filter.K; 
AVE_velRMSE1=AVE_velRMSE1/filter.K; 
AVE_posRMSE2=AVE_posRMSE2/filter.K; 
AVE_velRMSE2=AVE_velRMSE2/filter.K; 

fprintf('AVE_posRMSE1=%f\n',AVE_posRMSE1);
fprintf('AVE_velRMSE1=%f\n',AVE_velRMSE1);
fprintf('AVE_posRMSE2=%f\n',AVE_posRMSE2);
fprintf('AVE_velRMSE2=%f\n',AVE_velRMSE2);

%% 画图
figure(1)
plot(filter.truth_X(1,:),filter.truth_X(3,:),'k-.');hold on
plot(est_CKF12(1,:),est_CKF12(3,:),'b-*');
plot(est_CKF22(1,:),est_CKF22(3,:),'r-*');
legend('真实航迹','estimation position', 'estimation velocity')
xlabel('x轴/m','fontsize',12)
ylabel('y轴/m','fontsize',12)

figure; 
subplot(311);
xlabel('x轴/m'); ylabel('y轴/m');
plot(filter.truth_X(1,:),filter.truth_X(3,:),'k-.');hold on;
plot(filter.meas1(2,:).*cos(filter.meas1(1,:)),filter.meas1(2,:).*sin(filter.meas1(1,:)),'ro');
plot(filter.meas2(2,:).*cos(filter.meas2(1,:)),filter.meas2(2,:).*sin(filter.meas2(1,:)),'b*');
% plot(est_CKF12(1,:),est_CKF12(3,:),'b-*');
% plot(est_CKF22(1,:),est_CKF22(3,:),'r-*');
legend('真实航迹','sensor1测量数据','sensor2测量数据','Location','northwest'); grid on; grid minor; % 'localfilter1 CKF估计结果','localfilter2 CKF估计结果'

subplot(312);
plot(1:filter.K,RMSE_posCKF1,'b.-','LineWidth',1.5);hold on;
plot(1:filter.K,RMSE_posCKF2,'r.-','LineWidth',1.5);
xlabel('采样时刻/s'); ylabel('估计RMSE/m');grid on; grid minor; 
legend('localfilter1 CKF位置RMSE','localfilter1 CKF位置RMSE','Location','northwest');

subplot(313);
plot(1:filter.K,RMSE_velCKF1,'b.-','LineWidth',1.5);hold on;
plot(1:filter.K,RMSE_velCKF2,'r.-','LineWidth',1.5);
xlabel('采样时刻/s'); ylabel('估计RMSE/m');grid on; grid minor; 
legend('localfilter1 CKF速度RMSE','localfilter2 CKF速度RMSE','Location','northwest');