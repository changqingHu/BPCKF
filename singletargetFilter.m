classdef singletargetFilter
    properties
        K; % �۲���֡��
        T; % �˲����
        % ====== ��̬ģ�Ͳ��� =======
        targetStateDim; % ״̬ά��
        sigma_process;  % ��������Э����   
        processNoiseDim;% ��������ά��
        Q;              % ����������������
        % ====== ����ģ�Ͳ��� =======
        sigma_measNoise1;% ������1��������Э����
        sigma_measNoise2;% ������2��������Э����
        sigma_measNoise3;% ������3��������Э����
        v1;
        v2;
        v3;
        MeasNoiseDim;   % ��������ά��
        MeasDim;        % ��������ά��
        % ====== �������� =======
        truth_X;        % ��ʵ����
        meas1;          % ������1�Ĺ۲�����
        meas2;          % ������2�Ĺ۲�����
        meas3;          % ������3�Ĺ۲�����
        S1;             % ��ǰʱ�̹�������������������Э����
        S2;             % ��ǰʱ�̹�����������һʱ������������Э����     �������������ͬ�Ĵ�������ʹ��ͬһ�ײ���
    end
    
    
    methods
        function obj = singletargetFilter
            % ���캯��
            % ============== ������ʼ�� ===============
            obj.K = 50;
            obj.T = 1;
            obj.targetStateDim = 4;     % ״̬ά��
            obj.processNoiseDim = 2;    % ��������ά��
            obj.MeasNoiseDim = 2;       % ��������ά��
            obj.MeasDim = 2;            % ����ά��
            obj.sigma_process = diag([0.2, 0.5]);          % ��������Э����
            obj.sigma_measNoise1 = diag([0.2;0.4]);       % ������1����������Э����  [0.1;3.9]
            obj.sigma_measNoise2 = diag([0.1;0.3]);         % ������2����������Э����  [pi/90;1]
            obj.sigma_measNoise3 = diag([pi/90;5]);         % ������3����������Э����
            obj.Q = [obj.T^3/3 0 ; obj.T^2/2 0 ; 0 obj.T^2/2 ; 0 obj.T ];    % ����������������  ��ʱ���Ϊ tao_k ---> ����������ϵ��
            obj.S1 = [0.85, 2.54; 0.02, 1.21];   % ��ǰʱ�̹�������������������Э���� 3x2�ľ���     [0.01, 0.1; 0.01, 0.01; 0.01, 0.01]; 34
            obj.S2 = [0.05, 0.05; 0.01, 0.01];   % ��ǰʱ�̹�����������һʱ������������Э���� 3x2�ľ���
        end
        % ============== ���ɶ�ά�����͹۲� ===============
        function obj = gen_model(obj)
            target_state  = [ 0; 5; 1; 0];
            randn('seed',0);
            for k = 1:obj.K
                target_state = obj.CT_dynamin_model(obj.T)*target_state;
                obj.truth_X(:,k) = target_state;                                                        % �켣��ֵ
                % ����
                obj.meas1(:,k) = [atan2(target_state(3,:),target_state(1,:)); 
                                sqrt(sum(target_state([1 3],:).^2,1))]...
                    +obj.sigma_measNoise1 * randn(obj.MeasNoiseDim,size(target_state,2));               % ������1����������
                obj.v1(:,k) = obj.sigma_measNoise1 * randn(obj.MeasNoiseDim,size(target_state,2));      % ������1����������
                
                
                obj.meas2(:,k) = [atan2(target_state(3,:),target_state(1,:)); 
                                sqrt(sum(target_state([1 3],:).^2,1))]...
                    +obj.sigma_measNoise2 * randn(obj.MeasNoiseDim,size(target_state,2));               % ������2����������
                obj.v2(:,k) = obj.sigma_measNoise2 * randn(obj.MeasNoiseDim,size(target_state,2));      % ������2����������
                
                obj.meas3(:,k) = [atan2(target_state(3,:),target_state(1,:)); 
                                sqrt(sum(target_state([1 3],:).^2,1))]...
                    +obj.sigma_measNoise3 * randn(obj.MeasNoiseDim,size(target_state,2));               % ������3����������
                obj.v3(:,k) = obj.sigma_measNoise3 * randn(obj.MeasNoiseDim,size(target_state,2));      % ������3����������
            end
        end     
        %% ============== BP-CKF ��������״̬Ԥ�� ===============
        %��һ��״̬Ԥ��
        %========================================================================================%
        % step1 �����ݻ���   ���������ݻ��� δ�����ݻ���
        function [weight_CP,state_CP] = gen_cubaturePoint(obj,statePrior,covarPrior)     
            cubaturePointNum = 2*obj.targetStateDim;        % N = 2n = 2*5;
            varphi = sqrt(obj.targetStateDim)*cat(2,eye(obj.targetStateDim),-1*eye(obj.targetStateDim));  % cat(dim, A, B)�����������
           
            % Э������󷽸�
            choleskyDecompose = obj.cholPSD(covarPrior);            % cholesky�ֽ�
            % cubature��״̬
            state_CP = zeros(obj.targetStateDim,cubaturePointNum);  % n*2n
            
            for iCubaturePoint = 1:cubaturePointNum
                state_CP(:,iCubaturePoint) = statePrior+choleskyDecompose*varphi(:,iCubaturePoint);
            end
            % cubature��Ȩ��
            weight_CP = 1/cubaturePointNum.*ones(1,cubaturePointNum);
        end
        
        
        % ��������������ݻ��������        statePrior = [x, v]
        function [weight_CP,xv] = gen_cubaturePoint_ZengGuang(obj,statePrior,covarPrior)
            cubaturePointNum = 2*(obj.targetStateDim + obj.MeasNoiseDim); % �ݻ���ά�� 2*��n+m��
            varphi = sqrt(obj.targetStateDim+obj.MeasNoiseDim)*cat(2,eye(obj.targetStateDim+obj.MeasNoiseDim),-1*eye(obj.targetStateDim+obj.MeasNoiseDim));  % cat(dim, A, B)�����������
            
            % Э������󷽸�
            choleskyDecompose = obj.cholPSD(covarPrior);            % cholesky�ֽ�
            % cubature��״̬
            xv = zeros(obj.targetStateDim+obj.MeasNoiseDim,cubaturePointNum);  % n*2n
            
            for iCubaturePoint = 1:cubaturePointNum
                xv(:,iCubaturePoint) = statePrior+choleskyDecompose*varphi(:,iCubaturePoint);
            end
            % cubature��Ȩ��
            weight_CP = 1/cubaturePointNum.*ones(1,cubaturePointNum);
        end
        
        
        %============================================ ״̬����Ԥ�� N(x(k);x(k|k-1),p(k|k-1)) ============================================%
        function [x1,N1] = State_two_steps_prediction(obj, statePrior,covarPrior)
            [weight_CP,state_CP] = obj.gen_cubaturePoint(statePrior,covarPrior);
            % cubature��״̬Ԥ��
            cubaturePoint_num = size(state_CP,2);   % ����state_CP �ĵڶ�����ά�� 2n
            CP1 = zeros(obj.targetStateDim,cubaturePoint_num);  % ά�� n*2n

            % �ݻ���ķ����Դ���
            for i = 1:cubaturePoint_num
                CP1(:,i) = obj.CT_dynamin_model(obj.T)*state_CP(:,i); % �ݻ���ķ����Դ���
            end
            
            % ��Ȩ���
            x1 = CP1*weight_CP';
            
            % Ԥ��Э����
            N1 = zeros(obj.targetStateDim,obj.targetStateDim);
            for i = 1:cubaturePoint_num
                N1 = N1+weight_CP(1,i)*CP1(:,i)*CP1(:,i)';
            end
%             p1 = p1+obj.Q*sqrtm(obj.sigma_process)*obj.Q'-x1*(x1)';
%             %   obj.Q*sqrtm(obj.sigma_process)*obj.Q'  
            N1 = N1 + obj.Q*(obj.sigma_process)*(obj.Q)' - x1*(x1)';
        end    
        
        %============================================ ״̬һ��Ԥ�� N(x(k);x1,p1) ============================================%
        %  ���룺 ��������Ԥ���״̬�����kʱ�̵Ĺ۲⣬ kʱ�̵�����Ԥ�⣬ kʱ�̵���Ϣ���kʱ�̵�״̬�������㼰����
        function [x2,p2] = State_one_steps_prediction(obj, x1, N1, zk, zk_pre_last, pzz_last, xv, N2)   % xv��������Ϊ0
            % �����ݻ��㼰��Ȩֵ
            [weight_CP_pre,state_CP_pre] = obj.gen_cubaturePoint(x1,N1);
            [weight_CP_xv,state_CP_xv] = obj.gen_cubaturePoint_ZengGuang(xv,N2);
           
            % �����ݻ���ķֽ�
            xcp = [state_CP_xv(1,:); state_CP_xv(2,:); state_CP_xv(3,:); state_CP_xv(4,:) ];      % ����������x���� 
            vcp = [state_CP_xv(5,:); state_CP_xv(6,:)];                                                             % ����������v����
            
            % cubature��״̬Ԥ��
            cubaturePoint_num = size(state_CP_pre,2);   % ����state_CP �ĵڶ�����ά�� 2n             5x10
            cubaturePoint_num_xv = size(state_CP_xv,2);   % ����state_CP �ĵڶ�����ά�� 2(n+m)       7x14
            CP2 = zeros(obj.targetStateDim,cubaturePoint_num);  % ά�� n*2n
%             CP2_fxv = zeros(obj.targetStateDim+obj.MeasNoiseDim,cubaturePoint_num);
%             fai_h = zeros([obj.targetStateDim,obj.MeasDim],cubaturePoint_num);
            
            % ����Ԥ��ֵ  �˴�Ҳ���������ݻ���ķ����Ա任 h(x)
            hx = [atan2(state_CP_pre(3,:),state_CP_pre(1,:)); sqrt(sum(state_CP_pre([1 3],:).^2,1))];   % �ؼ��� ��.�� ���վ���Ԫ������
            
            t1 = [0,0;0,0;0,0;0,0];
            t2 = [0,0;0,0;0,0;0,0];
            % �ݻ���ķ����Դ���
            for i = 1:cubaturePoint_num
                CP2(:,i) = obj.CT_dynamin_model(obj.T) * state_CP_pre(:,i); % �ݻ���ķ����Դ��� f(x)
%                 fai_h(:,i) = CP2(:,i)*hx(:,i)';                                               % fai(xk)*h(xk)'   
                t1 = t1 + weight_CP_pre(1,i)*CP2(:,i)*hx(:,i)';
            end
%             fprintf('hx=%f\n',hx);
            for j = 1:cubaturePoint_num_xv
%                 CP2_fxv(:,j) = obj.CT_dynamin_model(obj.T) * xcp(:,j) * vcp(:,j)';  % ���������� f(xk)*v(k)'  5x5 * 5x1 * 1x2 = 5x2
                t2 = t2 + weight_CP_xv(1,j)*obj.CT_dynamin_model(obj.T) * xcp(:,j) * vcp(:,j)';
            end
            
%             t1 = fai_h * weight_CP_pre';      % fai(xk)*h(xk)' �Ļ���
%             t2 = CP2_fxv*weight_CP_xv';       % f(xk)*v(k)'�Ļ���
            pxz = t1 + t2 - x1*zk_pre_last' + obj.Q*obj.S1;
%             fprintf('t1=%f\n',t1);
%             fprintf('t2=%f\n',t2);
            Mk = pxz/(pzz_last);            % ����������
%             fprintf('Mk=%f\n',Mk);
            x2 = x1 + Mk*(zk - zk_pre_last);
            p2 = N1 - Mk*pzz_last*Mk';
        end
    
        %============================================ ������� N(x(k);x2,p2) ============================================
        function [x3,p3,K_gain,zpre,N3,Pzz,Pxz] = measurement_update(obj, x2, p2, measurement, Rv1)
            % �����ݻ��㼰��Ȩֵ
            [weight_CP_pre,state_CP_pre] = obj.gen_cubaturePoint(x2,p2);
            
            % �������������ݻ��㼰��Ȩֵ
%             Rv1 = obj.sigma_measNoise1;     % ��һ����������������������
            pxv = obj.Q*(obj.S2);
            N3 = [p2, pxv; pxv', Rv1];
            xv = [x2; 0; 0];
            [weight_CP_xv,state_CP_xv] = obj.gen_cubaturePoint_ZengGuang(xv,N3);
            % �����ݻ���ķֽ�
            xcp = [state_CP_xv(1,:); state_CP_xv(2,:); state_CP_xv(3,:); state_CP_xv(4,:)];      % ����������x���� 
            vcp = [state_CP_xv(5,:); state_CP_xv(6,:)];                                                             % ����������v����
            
            % ����Ԥ��ֵ  �����ݻ���ķ����Ա任 h(x)   2x(2n)
            hx = [atan2(state_CP_pre(3,:),state_CP_pre(1,:)); sqrt(sum(state_CP_pre([1 3],:).^2,1))];   % �ؼ��� ��.�� ���վ���Ԫ������
            Hx = [atan2(xcp(3,:),xcp(1,:)); sqrt(sum(xcp([1 3],:).^2,1))];
            
            zpre = hx * weight_CP_pre';                                         % k+1ʱ�̵�����Ԥ��ֵ
%             pzz1 = hx * hx' * weight_CP_pre';                                 % hx*hx�Ļ��� 
%             fprintf('zpre=%f\n',zpre);
            sigmaPoint_num = size(state_CP_pre, 2);
            pzz1 = zeros(obj.MeasDim, obj.MeasDim);
            x_hx = zeros(obj.targetStateDim, obj.MeasDim);
            for j = 1:sigmaPoint_num
                pzz1 = pzz1 + weight_CP_pre(1,j)*hx(:,j)*hx(:,j)';
                x_hx = x_hx + weight_CP_pre(1,j)*state_CP_pre(:,j)*hx(:,j)';    % x*hx�Ļ���
            end
            Pxz = x_hx - (x2)*(zpre') + obj.Q*(obj.S2);                         % PxzЭ����
            
            
            sigmaPoint_num_xv = size(state_CP_xv,2);                            % ����������ά�� 
            pzz2 = zeros(obj.MeasDim, obj.MeasNoiseDim);
            for i = 1:sigmaPoint_num_xv
                pzz2 = pzz2 + weight_CP_xv(1,i)*Hx(:,i)*vcp(:,i)';
            end
%             fprintf('pzz1=%f\n',pzz1);
%             fprintf('pzz2=%f\n',pzz2);
            Pzz = pzz1 + pzz2 + (pzz2') + Rv1 - zpre*zpre';
            K_gain = Pxz/Pzz;        % ����������
            innovation  = measurement - zpre;
            x3 = x2 + K_gain*innovation;         
            p3 = p2 - K_gain*Pzz*(K_gain');
        end    
        
        %============================================ ���Կռ��� һ��״̬Ԥ�� ============================================
        % x p ��Ҫ��ʼ�� ���ó�ʼ״̬
        function [x4, p4] = LS_prediction(obj, x, p)
            [weight_CP_pre,state_CP_pre] = obj.gen_cubaturePoint(x,p);
            cp_num = size(state_CP_pre, 2);
            
            x = zeros(obj.targetStateDim,1);
            p = zeros(obj.targetStateDim, obj.targetStateDim);
            for i = 1: cp_num
            fx = obj.CT_dynamin_model(obj.T) * state_CP_pre(:,i);
            x = x + weight_CP_pre(1,i)*(fx);
            p = p + weight_CP_pre(1,i)*(fx)*(fx');
            end
            x4 = x;
            p4 = p - x*(x') + (obj.Q)*(obj.sigma_process)*(obj.Q)';
        end
        
        %============================================ ���Կռ��� ������� ============================================
        function [x5, p5] = LS_update(obj, x4, p4, x3, K_gain, Rv1)      % p4 = N4  x3Ϊ����
            [weight_CP,state_CP] = obj.gen_cubaturePoint(x4,p4);
            
%             Rv1 = obj.sigma_measNoise1;
            N5 = [p4, obj.Q*obj.S2; (obj.Q*obj.S2)', Rv1];
            xv = [x4; 0; 0];
            [weight_CP_xv,state_CP_xv] = obj.gen_cubaturePoint_ZengGuang(xv, N5);
            xcp = [state_CP_xv(1,:); state_CP_xv(2,:); state_CP_xv(3,:); state_CP_xv(4,:)];      % ����������x���� 
            vcp = [state_CP_xv(5,:); state_CP_xv(6,:)];                                          % ����������v����
            
            PXx = zeros(obj.targetStateDim, obj.MeasDim);
%             PXz = zeros(obj.targetStateDim, obj.targetStateDim);
            cp_num = size(state_CP,2);
            xv_cp_num = size(state_CP_xv,2);
            hx = [atan2(state_CP(3,:),state_CP(1,:)); sqrt(sum(state_CP([1 3],:).^2,1))];
            hx1 = [atan2(x4(3),x4(1)); sqrt((x4(1)^2+x4(3)^2))];

            G1 = zeros(obj.MeasDim, obj.MeasDim);
            G2 = zeros(obj.MeasDim, obj.MeasNoiseDim);
            for i = 1:cp_num
                PXx = PXx + weight_CP(1,i)*state_CP(:,i)*(hx(:,i) - hx1)';
                G1 = G1 + weight_CP(1,i)*(hx(:,i) - hx1)*(hx(:,i) - hx1)';
            end
            PXz = (PXx + obj.Q*obj.S2)*K_gain';         % ʽ22
            
            hx2 = [atan2(xcp(3,:),xcp(1,:)); sqrt(sum(xcp([1 3],:).^2,1))];
            
            for j = 1:xv_cp_num
                G2 = G2 + weight_CP_xv(1,j)*(hx2(:,j))*vcp(:,j)';
            end
            
            G = K_gain * (G1+G2+G2'+Rv1) * K_gain';
            L = PXz / G;
            
            x5 = x4 + L*(x3 - x4);
            p5 = p4 - L*G*L';
%             p5 = p4 - PXz*L' - (PXz*L')' + L*G*L';
            
        end
        
        
        
        
        
    end
        
        methods(Static)
        % ��̬ģ��
        % ============== ����ת��ģ�ͣ�CT model�� ===================
        function F = CT_dynamin_model(T)
            omega = 0.04;   % x_prior(5)        omegaΪ��ת��
            F = [1 sin(omega*T)/omega 0 -((1-cos(omega*T))/omega) ;
                0 cos(omega*T) 0 -sin(omega*T) ;
                0 (1-cos(omega*T))/omega 1 sin(omega*T)/omega ;
                0 sin(omega*T) 0 cos(omega*T) ];
        end
        
        % �۲�ģ��
        % ============== ����/�Ƕȹ۲� ===================
        % 
        % Jacobian����
        function H = CT_measurement_model(x)
            p = x([1 3],:);         % ȡ����x�ĵ�һ�к͵�����    �����������У�
            mag = p(1)^2 + p(2)^2;
            sqrt_mag = sqrt(mag);
            H = [-p(2)/mag  0  p(1)/mag 0  0 ; ...
                 p(1)/sqrt_mag  0  p(2)/sqrt_mag  0  0];
        end

        % ============== ����cholesky�ֽ� ==============
        function Xi = cholPSD(A)
%             [~, flag] = chol(A);
%             if (flag == 0)
                Xi = (chol(A))';
%             else
%                 [~,S,V] = svd(A);
%                 Ss = sqrt(S);
%                 Xi = V*Ss;
%             end
        end    
       end
    
end