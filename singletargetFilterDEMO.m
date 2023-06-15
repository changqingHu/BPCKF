clear; close all; clc
randn('seed',0);
filter = singletargetFilter;
filter = filter.gen_model;

MCRuns = 1;     %���ؿ������д���

% ��ʼ��
RMSE_posCKF1 = zeros(MCRuns,filter.K); % �˲���1��λ�þ��������
RMSE_velCKF1 = zeros(MCRuns,filter.K); % �˲���1���ٶȾ��������
RMSE_posCKF2 = zeros(MCRuns,filter.K); % �˲���2��λ�þ��������
RMSE_velCKF2 = zeros(MCRuns,filter.K); % �˲���2���ٶȾ��������
AVE_posRMSE1=0; 
AVE_velRMSE1=0; 
AVE_posRMSE2=0; 
AVE_velRMSE2=0;
for iMCruns = 1:MCRuns

    stateUpd_CKF1 = [ 1.5; 1.0; 0.21; 0.64];                 % �˲���1��״̬��ʼֵ   [ 1.5; 1.0; 0.21; 0.64]
    covarUpd_CKF1 = blkdiag(1.2*eye(3),0.5);                 % �˲���1��״̬�����ʼֵ  blkdiag(10*eye(4),pi/90); (10,10,100,10,0.1)
    stateUpd_CKF2 = [ 1.6; 1.0; 0.11; 0.64];                 % �˲���2��״̬��ʼֵ   [ 0; 6; 0; 1; 0.02 ];
    covarUpd_CKF2 = blkdiag(1*eye(3),pi/90);                 % �˲���2��״̬�����ʼֵ  blkdiag(10*eye(4),pi/90); (10,10,100,10,0.1)
    est_CKF11 = zeros(filter.targetStateDim,filter.K);       % ��1���˲����ĵ�1��״̬���ƴ洢����
    est_CKF12 = zeros(filter.targetStateDim,filter.K);       % ��1���˲����ĵ�2��״̬���ƴ洢����
    est_CKF21 = zeros(filter.targetStateDim,filter.K);       % ��2���˲����ĵ�1��״̬���ƴ洢����
    est_CKF22 = zeros(filter.targetStateDim,filter.K);       % ��2���˲����ĵ�2��״̬���ƴ洢����
    xpre1 = zeros(filter.targetStateDim,filter.K);           % ��1���˲����ĵ�1���˲���һ��Ԥ��״̬����ֵ�洢
    xpre2 = zeros(filter.targetStateDim,filter.K);           % ��2���˲����ĵ�1���˲���һ��Ԥ��״̬����ֵ�洢
    est_CKF_final = zeros(filter.targetStateDim,filter.K);   % �˲�������BP�ںϺ��״̬���ƴ洢����

    tCKF = 0;
    est_CKF11(:,1) = filter.truth_X(:,1);
    est_CKF12(:,1) = filter.truth_X(:,1);
    for k = 2:filter.K
       %%
        tic
        %================================ ��һ���˲��� %=========================
        % ��һ������������״̬Ԥ��
        [x1, p1] = filter.State_two_steps_prediction(stateUpd_CKF1, covarUpd_CKF1);
        
        % ��һ��������һ��״̬Ԥ��
        zk = filter.meas1(:,k-1);
        if (k==2)
            zk_pre_last = filter.meas1(:,1);                     % ����ʵ������Ϊ��ʼʱ�̵�����Ԥ�� ��Ҫ�ں������и���  [0.2155; 6.7194]
            xpre1(:,k-1) = filter.truth_X(:,1);                  % ����ʵ״̬��Ϊ��ʼʱ�̵�״̬Ԥ�� ��Ҫ���и���[5.9; 5.9; 1.0; 0.12; 0.02]
            p2 = blkdiag(0.25, 0.25 ,0.05 ,0.01);                 % ������״̬Ԥ�ⷽ����Ϊ��һ��״̬Ԥ�ⷽ�� blkdiag(10*eye(5))  p1
            pzz_last = blkdiag(1.6,3.6);                         % 10,pi/90
            pxv = filter.Q*(filter.S2);                          % ���������ʱ�ļ��㷽ʽһ��
            N2 = [p2, pxv; pxv', filter.sigma_measNoise1];
        end
        xv = [xpre1(:,k-1); 0; 0 ];
        [x2,p2] = filter.State_one_steps_prediction(x1, p1, zk, zk_pre_last, pzz_last, xv, N2);
        xpre1(:,k) = x2;
        
        % ��һ���������������
        measurement = filter.meas1(:,k);                        % ��ϵ�ǰʱ�̵��������ݽ��и���
        Rv1 = filter.sigma_measNoise1;                          % ������1��������������
        [x3,p3,K_gain,zpre, N3,Pzz,Pxz] = filter.measurement_update(x2, p2, measurement, Rv1);
        zk_pre_last = zpre;
        N2 = N3;
        pzz_last = Pzz;
        est_CKF11(:,k) = x3;
%         stateUpd_CKF1 = x3;
%         covarUpd_CKF1 = p3;
        
        %================================ �ڶ����˲��� %=========================
        % ��һ�������������Կռ��µ�һ��״̬Ԥ��
        if (k==2)
            x_final = filter.truth_X(:,1);      % x3
            p_final = blkdiag(0.55, 0.04 ,0.04 ,0.09);      % blkdiag(0.1*eye(5))    1.05, 0.40 ,0.54 ,0.09
        end
        [x4, p4] = filter.LS_prediction(x_final, p_final);
        
        % ��һ�������������Կռ��µ��������
%         test = [p4, filter.Q*filter.S2; (filter.Q*filter.S2)', filter.sigma_measNoise1];
        [x_final, p_final] = filter.LS_update(x4, p4, x3, K_gain, Rv1);
        stateUpd_CKF1 = x_final;     % �����˲����״̬��������
        covarUpd_CKF1 = p_final;     % �����˲���ķ����������
        est_CKF12(:,k) = x_final;    % �����˲���״̬�Ĵ洢
        
        %xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx �ڶ�����������ʼ xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        %================================ ��һ���˲��� %=========================
        % �ڶ�������������״̬Ԥ��
        [x21, p21] = filter.State_two_steps_prediction(stateUpd_CKF2, covarUpd_CKF2);
        
        % ��һ��������һ��״̬Ԥ��
        zk2 = filter.meas2(:,k-1);
        if (k==2)
            zk_pre_last2 = filter.meas2(:,1);               % ����ʵ������Ϊ��ʼʱ�̵�����Ԥ�� ��Ҫ�ں������и��� [0.15; 7.78]
            xpre2(:,k-1) = filter.truth_X(:,1);             % ����ʵ״̬��Ϊ��ʼʱ�̵�״̬Ԥ�� ��Ҫ���и��� [5.9; 5.9; 1.0; 0.12; 0.02]
            p22 = blkdiag(1.05, 0.55 ,1.04 ,1.1);    % ������״̬Ԥ�ⷽ����Ϊ��һ��״̬Ԥ�ⷽ��  p21 0.001*eye(5)
            pzz_last2 = blkdiag(1.8,3.6);                      % blkdiag(1,pi/90)
            pxv2 = filter.Q*(filter.S2);                    % ���������ʱ�ļ��㷽ʽһ��
            N22 = [p22, pxv2; pxv2', filter.sigma_measNoise1];
        end
        xv2 = [xpre2(:,k-1); 0; 0 ];
          
        [x22,p22] = filter.State_one_steps_prediction(x21, p21, zk2, zk_pre_last2, pzz_last2, xv2, N22);
        xpre2(:,k) = x22;
        
        % ��һ���������������
        measurement2 = filter.meas2(:,k);  % ��ϵ�ǰʱ�̵��������ݽ��и���
        Rv2 = filter.sigma_measNoise2;     % ������2��������������
        [x23,p23,K_gain2,zpre2, N23,Pzz2,Pxz2] = filter.measurement_update(x22, p22, measurement2, Rv2);
        zk_pre_last2 = zpre2;
        N22 = N23;
        pzz_last2 = Pzz2;
        est_CKF21(:,k) = x23;
%         stateUpd_CKF2 = x23;     % �����˲����״̬��������
%         covarUpd_CKF2 = p23;     % �����˲���ķ����������
        
        %================================ �ڶ����˲��� %=========================
        % ��һ�������������Կռ��µ�һ��״̬Ԥ��
        if (k==2)
            x_final2 = filter.truth_X(:,1);                     % x23
            p_final2 = blkdiag(1.05, 0.29 ,0.24 ,0.49);         % p23
        end
        [x24, p24] = filter.LS_prediction(x_final2, p_final2);
        
        % ��һ�������������Կռ��µ��������
        [x_final2, p_final2] = filter.LS_update(x24, p24, x23, K_gain2, Rv2);
        stateUpd_CKF2 = x_final2;     % �����˲����״̬��������
        covarUpd_CKF2 = p_final2;     % �����˲���ķ����������
        est_CKF22(:,k) = x_final2;    % �����˲���״̬�Ĵ洢
        
        
        % �˲���1״̬RMSE
        RMSE_posCKF1(iMCruns,k) = sqrt(sum((stateUpd_CKF1([1 3])-filter.truth_X([1 3],k)).^2));
        RMSE_velCKF1(iMCruns,k) = sqrt(sum((stateUpd_CKF1([2 4])-filter.truth_X([2 4],k)).^2));
        % �˲���2״̬RMSE
        RMSE_posCKF2(iMCruns,k) = sqrt(sum((stateUpd_CKF2([1 3])-filter.truth_X([1 3],k)).^2));
        RMSE_velCKF2(iMCruns,k) = sqrt(sum((stateUpd_CKF2([2 4])-filter.truth_X([2 4],k)).^2));
        
        tCKF = tCKF+toc;
        fprintf('K=%f\n',k);
        %%
    end
    
    %    disp(['Current Iteration is ',num2str(iMCruns),'.']);
    %
    disp('========================');
    disp('�ķ�ʱ��/s��');
    disp(['CKF:',num2str(tCKF)]);
end
% delete(h);

RMSE_posCKF1 = mean(RMSE_posCKF1,1); RMSE_velCKF1 = mean(RMSE_velCKF1,1);     % ��ȡ��ÿһ�еľ�ֵ
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

%% ��ͼ
figure(1)
plot(filter.truth_X(1,:),filter.truth_X(3,:),'k-.');hold on
plot(est_CKF12(1,:),est_CKF12(3,:),'b-*');
plot(est_CKF22(1,:),est_CKF22(3,:),'r-*');
legend('��ʵ����','estimation position', 'estimation velocity')
xlabel('x��/m','fontsize',12)
ylabel('y��/m','fontsize',12)

figure; 
subplot(311);
xlabel('x��/m'); ylabel('y��/m');
plot(filter.truth_X(1,:),filter.truth_X(3,:),'k-.');hold on;
plot(filter.meas1(2,:).*cos(filter.meas1(1,:)),filter.meas1(2,:).*sin(filter.meas1(1,:)),'ro');
plot(filter.meas2(2,:).*cos(filter.meas2(1,:)),filter.meas2(2,:).*sin(filter.meas2(1,:)),'b*');
% plot(est_CKF12(1,:),est_CKF12(3,:),'b-*');
% plot(est_CKF22(1,:),est_CKF22(3,:),'r-*');
legend('��ʵ����','sensor1��������','sensor2��������','Location','northwest'); grid on; grid minor; % 'localfilter1 CKF���ƽ��','localfilter2 CKF���ƽ��'

subplot(312);
plot(1:filter.K,RMSE_posCKF1,'b.-','LineWidth',1.5);hold on;
plot(1:filter.K,RMSE_posCKF2,'r.-','LineWidth',1.5);
xlabel('����ʱ��/s'); ylabel('����RMSE/m');grid on; grid minor; 
legend('localfilter1 CKFλ��RMSE','localfilter1 CKFλ��RMSE','Location','northwest');

subplot(313);
plot(1:filter.K,RMSE_velCKF1,'b.-','LineWidth',1.5);hold on;
plot(1:filter.K,RMSE_velCKF2,'r.-','LineWidth',1.5);
xlabel('����ʱ��/s'); ylabel('����RMSE/m');grid on; grid minor; 
legend('localfilter1 CKF�ٶ�RMSE','localfilter2 CKF�ٶ�RMSE','Location','northwest');