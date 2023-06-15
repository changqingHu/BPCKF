classdef singletargetFilter
    properties
        K; % 观测总帧数
        T; % 滤波间隔
        % ====== 动态模型参数 =======
        targetStateDim; % 状态维数
        sigma_process;  % 过程噪声协方差   
        processNoiseDim;% 过程噪声维数
        Q;              % 过程噪声驱动矩阵
        % ====== 测量模型参数 =======
        sigma_measNoise1;% 传感器1测量噪声协方差
        sigma_measNoise2;% 传感器2测量噪声协方差
        sigma_measNoise3;% 传感器3测量噪声协方差
        v1;
        v2;
        v3;
        MeasNoiseDim;   % 测量噪声维数
        MeasDim;        % 测量数据维数
        % ====== 交互参数 =======
        truth_X;        % 真实航迹
        meas1;          % 传感器1的观测数据
        meas2;          % 传感器2的观测数据
        meas3;          % 传感器3的观测数据
        S1;             % 当前时刻过程噪声与量测噪声的协方差
        S2;             % 当前时刻过程噪声与上一时刻量测噪声的协方差     假设存在三个相同的传感器，使用同一套参数
    end
    
    
    methods
        function obj = singletargetFilter
            % 构造函数
            % ============== 参数初始化 ===============
            obj.K = 50;
            obj.T = 1;
            obj.targetStateDim = 4;     % 状态维度
            obj.processNoiseDim = 2;    % 过程噪声维度
            obj.MeasNoiseDim = 2;       % 量测噪声维度
            obj.MeasDim = 2;            % 量测维度
            obj.sigma_process = diag([0.2, 0.5]);          % 过程噪声协方差
            obj.sigma_measNoise1 = diag([0.2;0.4]);       % 传感器1的量测噪声协方差  [0.1;3.9]
            obj.sigma_measNoise2 = diag([0.1;0.3]);         % 传感器2的量测噪声协方差  [pi/90;1]
            obj.sigma_measNoise3 = diag([pi/90;5]);         % 传感器3的量测噪声协方差
            obj.Q = [obj.T^3/3 0 ; obj.T^2/2 0 ; 0 obj.T^2/2 ; 0 obj.T ];    % 过程噪声驱动矩阵  暂时理解为 tao_k ---> 过程噪声的系数
            obj.S1 = [0.85, 2.54; 0.02, 1.21];   % 当前时刻过程噪声与量测噪声的协方差 3x2的矩阵     [0.01, 0.1; 0.01, 0.01; 0.01, 0.01]; 34
            obj.S2 = [0.05, 0.05; 0.01, 0.01];   % 当前时刻过程噪声与上一时刻量测噪声的协方差 3x2的矩阵
        end
        % ============== 生成二维航迹和观测 ===============
        function obj = gen_model(obj)
            target_state  = [ 0; 5; 1; 0];
            randn('seed',0);
            for k = 1:obj.K
                target_state = obj.CT_dynamin_model(obj.T)*target_state;
                obj.truth_X(:,k) = target_state;                                                        % 轨迹真值
                % 测量
                obj.meas1(:,k) = [atan2(target_state(3,:),target_state(1,:)); 
                                sqrt(sum(target_state([1 3],:).^2,1))]...
                    +obj.sigma_measNoise1 * randn(obj.MeasNoiseDim,size(target_state,2));               % 传感器1的量测数据
                obj.v1(:,k) = obj.sigma_measNoise1 * randn(obj.MeasNoiseDim,size(target_state,2));      % 传感器1的量测噪声
                
                
                obj.meas2(:,k) = [atan2(target_state(3,:),target_state(1,:)); 
                                sqrt(sum(target_state([1 3],:).^2,1))]...
                    +obj.sigma_measNoise2 * randn(obj.MeasNoiseDim,size(target_state,2));               % 传感器2的量测数据
                obj.v2(:,k) = obj.sigma_measNoise2 * randn(obj.MeasNoiseDim,size(target_state,2));      % 传感器2的量测噪声
                
                obj.meas3(:,k) = [atan2(target_state(3,:),target_state(1,:)); 
                                sqrt(sum(target_state([1 3],:).^2,1))]...
                    +obj.sigma_measNoise3 * randn(obj.MeasNoiseDim,size(target_state,2));               % 传感器3的量测数据
                obj.v3(:,k) = obj.sigma_measNoise3 * randn(obj.MeasNoiseDim,size(target_state,2));      % 传感器3的量测噪声
            end
        end     
        %% ============== BP-CKF 包含二级状态预测 ===============
        %第一级状态预测
        %========================================================================================%
        % step1 生成容积点   仅仅生成容积点 未传播容积点
        function [weight_CP,state_CP] = gen_cubaturePoint(obj,statePrior,covarPrior)     
            cubaturePointNum = 2*obj.targetStateDim;        % N = 2n = 2*5;
            varphi = sqrt(obj.targetStateDim)*cat(2,eye(obj.targetStateDim),-1*eye(obj.targetStateDim));  % cat(dim, A, B)构造增广矩阵
           
            % 协方差矩阵方根
            choleskyDecompose = obj.cholPSD(covarPrior);            % cholesky分解
            % cubature点状态
            state_CP = zeros(obj.targetStateDim,cubaturePointNum);  % n*2n
            
            for iCubaturePoint = 1:cubaturePointNum
                state_CP(:,iCubaturePoint) = statePrior+choleskyDecompose*varphi(:,iCubaturePoint);
            end
            % cubature点权重
            weight_CP = 1/cubaturePointNum.*ones(1,cubaturePointNum);
        end
        
        
        % 针对增广矩阵进行容积点的生成        statePrior = [x, v]
        function [weight_CP,xv] = gen_cubaturePoint_ZengGuang(obj,statePrior,covarPrior)
            cubaturePointNum = 2*(obj.targetStateDim + obj.MeasNoiseDim); % 容积点维度 2*（n+m）
            varphi = sqrt(obj.targetStateDim+obj.MeasNoiseDim)*cat(2,eye(obj.targetStateDim+obj.MeasNoiseDim),-1*eye(obj.targetStateDim+obj.MeasNoiseDim));  % cat(dim, A, B)构造增广矩阵
            
            % 协方差矩阵方根
            choleskyDecompose = obj.cholPSD(covarPrior);            % cholesky分解
            % cubature点状态
            xv = zeros(obj.targetStateDim+obj.MeasNoiseDim,cubaturePointNum);  % n*2n
            
            for iCubaturePoint = 1:cubaturePointNum
                xv(:,iCubaturePoint) = statePrior+choleskyDecompose*varphi(:,iCubaturePoint);
            end
            % cubature点权重
            weight_CP = 1/cubaturePointNum.*ones(1,cubaturePointNum);
        end
        
        
        %============================================ 状态两步预测 N(x(k);x(k|k-1),p(k|k-1)) ============================================%
        function [x1,N1] = State_two_steps_prediction(obj, statePrior,covarPrior)
            [weight_CP,state_CP] = obj.gen_cubaturePoint(statePrior,covarPrior);
            % cubature点状态预测
            cubaturePoint_num = size(state_CP,2);   % 向量state_CP 的第二部分维度 2n
            CP1 = zeros(obj.targetStateDim,cubaturePoint_num);  % 维度 n*2n

            % 容积点的非线性传播
            for i = 1:cubaturePoint_num
                CP1(:,i) = obj.CT_dynamin_model(obj.T)*state_CP(:,i); % 容积点的非线性传播
            end
            
            % 加权求和
            x1 = CP1*weight_CP';
            
            % 预测协方差
            N1 = zeros(obj.targetStateDim,obj.targetStateDim);
            for i = 1:cubaturePoint_num
                N1 = N1+weight_CP(1,i)*CP1(:,i)*CP1(:,i)';
            end
%             p1 = p1+obj.Q*sqrtm(obj.sigma_process)*obj.Q'-x1*(x1)';
%             %   obj.Q*sqrtm(obj.sigma_process)*obj.Q'  
            N1 = N1 + obj.Q*(obj.sigma_process)*(obj.Q)' - x1*(x1)';
        end    
        
        %============================================ 状态一步预测 N(x(k);x1,p1) ============================================%
        %  输入： 对象，两步预测的状态及方差，k时刻的观测， k时刻的量测预测， k时刻的信息方差，k时刻的状态噪声增广及方差
        function [x2,p2] = State_one_steps_prediction(obj, x1, N1, zk, zk_pre_last, pzz_last, xv, N2)   % xv后面两项为0
            % 生成容积点及其权值
            [weight_CP_pre,state_CP_pre] = obj.gen_cubaturePoint(x1,N1);
            [weight_CP_xv,state_CP_xv] = obj.gen_cubaturePoint_ZengGuang(xv,N2);
           
            % 增广容积点的分解
            xcp = [state_CP_xv(1,:); state_CP_xv(2,:); state_CP_xv(3,:); state_CP_xv(4,:) ];      % 增广向量中x部分 
            vcp = [state_CP_xv(5,:); state_CP_xv(6,:)];                                                             % 增广向量中v部分
            
            % cubature点状态预测
            cubaturePoint_num = size(state_CP_pre,2);   % 向量state_CP 的第二部分维度 2n             5x10
            cubaturePoint_num_xv = size(state_CP_xv,2);   % 向量state_CP 的第二部分维度 2(n+m)       7x14
            CP2 = zeros(obj.targetStateDim,cubaturePoint_num);  % 维度 n*2n
%             CP2_fxv = zeros(obj.targetStateDim+obj.MeasNoiseDim,cubaturePoint_num);
%             fai_h = zeros([obj.targetStateDim,obj.MeasDim],cubaturePoint_num);
            
            % 量测预测值  此处也属于量测容积点的非线性变换 h(x)
            hx = [atan2(state_CP_pre(3,:),state_CP_pre(1,:)); sqrt(sum(state_CP_pre([1 3],:).^2,1))];   % 关键在 “.” 按照矩阵元素求幂
            
            t1 = [0,0;0,0;0,0;0,0];
            t2 = [0,0;0,0;0,0;0,0];
            % 容积点的非线性传播
            for i = 1:cubaturePoint_num
                CP2(:,i) = obj.CT_dynamin_model(obj.T) * state_CP_pre(:,i); % 容积点的非线性传播 f(x)
%                 fai_h(:,i) = CP2(:,i)*hx(:,i)';                                               % fai(xk)*h(xk)'   
                t1 = t1 + weight_CP_pre(1,i)*CP2(:,i)*hx(:,i)';
            end
%             fprintf('hx=%f\n',hx);
            for j = 1:cubaturePoint_num_xv
%                 CP2_fxv(:,j) = obj.CT_dynamin_model(obj.T) * xcp(:,j) * vcp(:,j)';  % 增广向量中 f(xk)*v(k)'  5x5 * 5x1 * 1x2 = 5x2
                t2 = t2 + weight_CP_xv(1,j)*obj.CT_dynamin_model(obj.T) * xcp(:,j) * vcp(:,j)';
            end
            
%             t1 = fai_h * weight_CP_pre';      % fai(xk)*h(xk)' 的积分
%             t2 = CP2_fxv*weight_CP_xv';       % f(xk)*v(k)'的积分
            pxz = t1 + t2 - x1*zk_pre_last' + obj.Q*obj.S1;
%             fprintf('t1=%f\n',t1);
%             fprintf('t2=%f\n',t2);
            Mk = pxz/(pzz_last);            % 卡尔曼增益
%             fprintf('Mk=%f\n',Mk);
            x2 = x1 + Mk*(zk - zk_pre_last);
            p2 = N1 - Mk*pzz_last*Mk';
        end
    
        %============================================ 量测更新 N(x(k);x2,p2) ============================================
        function [x3,p3,K_gain,zpre,N3,Pzz,Pxz] = measurement_update(obj, x2, p2, measurement, Rv1)
            % 生成容积点及其权值
            [weight_CP_pre,state_CP_pre] = obj.gen_cubaturePoint(x2,p2);
            
            % 生成增广向量容积点及其权值
%             Rv1 = obj.sigma_measNoise1;     % 第一个传感器的量测噪声方差
            pxv = obj.Q*(obj.S2);
            N3 = [p2, pxv; pxv', Rv1];
            xv = [x2; 0; 0];
            [weight_CP_xv,state_CP_xv] = obj.gen_cubaturePoint_ZengGuang(xv,N3);
            % 增广容积点的分解
            xcp = [state_CP_xv(1,:); state_CP_xv(2,:); state_CP_xv(3,:); state_CP_xv(4,:)];      % 增广向量中x部分 
            vcp = [state_CP_xv(5,:); state_CP_xv(6,:)];                                                             % 增广向量中v部分
            
            % 量测预测值  量测容积点的非线性变换 h(x)   2x(2n)
            hx = [atan2(state_CP_pre(3,:),state_CP_pre(1,:)); sqrt(sum(state_CP_pre([1 3],:).^2,1))];   % 关键在 “.” 按照矩阵元素求幂
            Hx = [atan2(xcp(3,:),xcp(1,:)); sqrt(sum(xcp([1 3],:).^2,1))];
            
            zpre = hx * weight_CP_pre';                                         % k+1时刻的量测预测值
%             pzz1 = hx * hx' * weight_CP_pre';                                 % hx*hx的积分 
%             fprintf('zpre=%f\n',zpre);
            sigmaPoint_num = size(state_CP_pre, 2);
            pzz1 = zeros(obj.MeasDim, obj.MeasDim);
            x_hx = zeros(obj.targetStateDim, obj.MeasDim);
            for j = 1:sigmaPoint_num
                pzz1 = pzz1 + weight_CP_pre(1,j)*hx(:,j)*hx(:,j)';
                x_hx = x_hx + weight_CP_pre(1,j)*state_CP_pre(:,j)*hx(:,j)';    % x*hx的积分
            end
            Pxz = x_hx - (x2)*(zpre') + obj.Q*(obj.S2);                         % Pxz协方差
            
            
            sigmaPoint_num_xv = size(state_CP_xv,2);                            % 增广向量的维度 
            pzz2 = zeros(obj.MeasDim, obj.MeasNoiseDim);
            for i = 1:sigmaPoint_num_xv
                pzz2 = pzz2 + weight_CP_xv(1,i)*Hx(:,i)*vcp(:,i)';
            end
%             fprintf('pzz1=%f\n',pzz1);
%             fprintf('pzz2=%f\n',pzz2);
            Pzz = pzz1 + pzz2 + (pzz2') + Rv1 - zpre*zpre';
            K_gain = Pxz/Pzz;        % 卡尔曼增益
            innovation  = measurement - zpre;
            x3 = x2 + K_gain*innovation;         
            p3 = p2 - K_gain*Pzz*(K_gain');
        end    
        
        %============================================ 线性空间下 一步状态预测 ============================================
        % x p 需要初始化 设置初始状态
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
        
        %============================================ 线性空间下 量测更新 ============================================
        function [x5, p5] = LS_update(obj, x4, p4, x3, K_gain, Rv1)      % p4 = N4  x3为量测
            [weight_CP,state_CP] = obj.gen_cubaturePoint(x4,p4);
            
%             Rv1 = obj.sigma_measNoise1;
            N5 = [p4, obj.Q*obj.S2; (obj.Q*obj.S2)', Rv1];
            xv = [x4; 0; 0];
            [weight_CP_xv,state_CP_xv] = obj.gen_cubaturePoint_ZengGuang(xv, N5);
            xcp = [state_CP_xv(1,:); state_CP_xv(2,:); state_CP_xv(3,:); state_CP_xv(4,:)];      % 增广向量中x部分 
            vcp = [state_CP_xv(5,:); state_CP_xv(6,:)];                                          % 增广向量中v部分
            
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
            PXz = (PXx + obj.Q*obj.S2)*K_gain';         % 式22
            
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
        % 动态模型
        % ============== 匀速转弯模型（CT model） ===================
        function F = CT_dynamin_model(T)
            omega = 0.04;   % x_prior(5)        omega为常转率
            F = [1 sin(omega*T)/omega 0 -((1-cos(omega*T))/omega) ;
                0 cos(omega*T) 0 -sin(omega*T) ;
                0 (1-cos(omega*T))/omega 1 sin(omega*T)/omega ;
                0 sin(omega*T) 0 cos(omega*T) ];
        end
        
        % 观测模型
        % ============== 距离/角度观测 ===================
        % 
        % Jacobian矩阵
        function H = CT_measurement_model(x)
            p = x([1 3],:);         % 取向量x的第一行和第三行    （包含所有列）
            mag = p(1)^2 + p(2)^2;
            sqrt_mag = sqrt(mag);
            H = [-p(2)/mag  0  p(1)/mag 0  0 ; ...
                 p(1)/sqrt_mag  0  p(2)/sqrt_mag  0  0];
        end

        % ============== 矩阵cholesky分解 ==============
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