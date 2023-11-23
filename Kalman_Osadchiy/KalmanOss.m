close all
clear all

bDoGridSearch = true;

%load the data 
data = load('arduino_data1.mat');
z = data.data(2000:end);
DSfactor = 1;
z = resample(z,1,DSfactor);

% set params
Fs = 200/DSfactor;
dt = 1/Fs;

[Pzz,freq] = pwelch(z,2*Fs,Fs,4*Fs,Fs);
frange = find(freq>2 & freq < 50); 

figure
plot(freq(frange),Pzz(frange));
xlabel('frequency, Hz')
ylabel('v^2/Hz');
title('PSD');

% visually inspect the above and set the central frequency
f0 = 11.5;% Hz

% NOTE: inspect the phase difference histogram. In case its mean is not at
% zero, tune the central frequency f0

%extract the ground truth
b_fir = fir1(fix(Fs/1), [f0-1 f0+1]/(Fs/2));
x_fir_lagged = filter(b_fir,[1],z);
x_fir = [x_fir_lagged(fix(length(b_fir)/2)+1:end) zeros(1,fix(length(b_fir)/2))];
h_fir = hilbert(x_fir);
e_fir = abs(h_fir);
ph_fir = atan2(imag(h_fir),real(h_fir));

H = [1,0];
delta = 2.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(bDoGridSearch)
    
    disp('Doing grid search... \n')
    
    clear err_x err_e MAX_xcf_e LAG_xcf_e MAX_xcf_x LAG_xcf_x MSE_x MSE_e MSE_en A SIGMA f 
    
    a_i = 1; 
    for a_ = 0.99:0.001:1.01 %0.99999
        sigma_i = 1;
        for sigma_ = 0.01:0.02:0.2
            % set KF matrices
            F = a_*[cos(2*pi*f0*dt) -sin(2*pi*f0*dt);...
                    sin(2*pi*f0*dt)  cos(2*pi*f0*dt)];
            Q = sigma_^2*eye(2);
            R = delta^2*eye(1); 

            % estimated analytic signal
            [x_ y_,KK] = kalman_filter(z,F,H,Q,R);

            % estimated envelope
            e_ = sqrt(sum(x_.*x_,1));

            % abs MSE
            
            err_x(a_i,sigma_i) = sum((x_fir - x_(1,:)).^2)/length(x_fir);
            err_e(a_i,sigma_i) = sum((e_fir - e_).^2)/length(x_fir);

            % cross-corr
            [xcf_e,lags_e] =  crosscorr(e_fir',e_',     'NumLags', fix(Fs/2));
            [xcf_x,lags_x] =  crosscorr(x_fir',x_(1,:)','NumLags', fix(Fs/2));

            % relative mse
            mse_e = norm(e_fir-e_)/norm(e_fir);
            mse_x = norm(x_fir-x_(1,:))/norm(x_fir);
            e_no = e_*sum(e_.*e_fir)/sum(e_.*e_);
            mse_en = norm(e_fir-e_no)/norm(e_fir);
    
            % save envelope accuracy
            [max_xcf_e, lag_xcf_e] = max(xcf_e);
            MAX_xcf_e(a_i,sigma_i) = max_xcf_e;
            LAG_xcf_e(a_i,sigma_i) = lags_e(lag_xcf_e);
            MSE_e(a_i,sigma_i)     = mse_e;
            MSE_en(a_i,sigma_i)    = mse_en;

            % save raw accruacy
            [max_xcf_x, lag_xcf_x] = max(xcf_x);
            MAX_xcf_x(a_i,sigma_i) = max_xcf_x;
            LAG_xcf_x(a_i,sigma_i) = lags_x(lag_xcf_x);
            MSE_x(a_i,sigma_i)     = mse_x;

            % save grid values    
            SIGMA(sigma_i) = sigma_;

            sigma_i = sigma_i+1;
        end;
        % save grid values
        A(a_i) = a_;
        a_i = a_i+1;
        [a_i,sigma_i]
    end;
    
    disp('Saving grid search results... \n')
    save MatsudaKFCFIRGridSearch_d2p5.mat err_x err_e MAX_xcf_e LAG_xcf_e ...
         MAX_xcf_x LAG_xcf_x MSE_x MSE_e MSE_en A SIGMA f0 
else
    disp('No grid search option chosen. Loading grid search results... \n')
    load MatsudaKFCFIRGridSearch.mat;
end;

%create cFIR filter
hb_cfir = hilbert(fir1(fix(Fs/8),[f0-3 f0+3]/(Fs/2)));

%filter the data
h_cfir = filter(hb_cfir,[1],z);

% compute envelope
e_cfir = abs(h_cfir);

% compute phase
ph_cfir = atan2(imag(h_cfir),real(h_cfir));

close all
fig_data = figure;
data_ax1 = subplot(4,3,1);
imagesc(SIGMA,A, MSE_en);
xlabel('sigma')
ylabel('a')
title('rel MSE envelope')
colorbar
subplot(4,3,2)
imagesc(SIGMA,A,(LAG_xcf_e));
xlabel('sigma')
ylabel('a')
title('LAG envelope')
colorbar
subplot(4,3,3)
imagesc(SIGMA,A,(MAX_xcf_e));
xlabel('sigma')
ylabel('a')
title('correlation envelope')
colorbar

subplot(4,3,4);
imagesc(SIGMA,A, MSE_x);
xlabel('sigma')
ylabel('a')
title('rel MSE raw')
colorbar
subplot(4,3,5)
imagesc(SIGMA,A,(LAG_xcf_x));
xlabel('sigma')
ylabel('a')
title('LAG raw')
colorbar
subplot(4,3,6)
imagesc(SIGMA,A,(MAX_xcf_x));
xlabel('sigma')
ylabel('a')
title('correlation raw')
colorbar

for it = 1:100
    axes(data_ax1);
    [sigma_,a_] = ginput(1);
    disp('sigma a')
    [sigma_, a_]

    clear x_pred x_ y_pred P_pred S K P_ y_ ;
    F = a_*[cos(2*pi*f0*dt) -sin(2*pi*f0*dt); sin(2*pi*f0*dt) cos(2*pi*f0*dt) ];
    Q = sigma_^2*eye(2);
    R = (delta)^2*eye(1); 

    [x_, y_, KK] = kalman_filter(z,F,H,Q,R);
    e_ = sqrt(sum(x_.*x_,1));

    err_x_this = sum((x_fir - x_(1,:)).^2)/length(x_fir);
    err_e_this = sum((e_fir - e_).^2)/length(x_fir);

    [xcf_e_this,lags_e_this]   =  crosscorr(e_fir',e_','NumLags', fix(Fs/2));
    [xcf_x_this,lags_x_this]   =  crosscorr(x_fir',x_(1,:)','NumLags', fix(Fs/2));
    
    [xcf_e_cfir,lags_e_cfir]   =  crosscorr(e_fir',e_cfir','NumLags', fix(Fs/2));
    
    
    % find max value and argmax of the cross-corr
    [max_xcf_e, lag_xcf_e]= max(xcf_e_this);
    [max_xcf_x, lag_xcf_x]= max(xcf_x_this);
    [max_xcf_e_cfir, lag_xcf_e_cfir]= max(xcf_e_cfir);
   
    
    % assess phase accruacy
    ph_est = atan2(x_(2,:),x_(1,:));
    ph_fir = atan2(imag(h_fir),real(h_fir));
    plv = norm(mean(exp(sqrt(-1)*(ph_fir-ph_est))));
    
    subplot(4,3,7)
    plot(lags_e_this, xcf_e_this);
    grid
    title(num2str([max_xcf_e, lags_e_this(lag_xcf_e)]))
    axis([-200 200 0 1])
    subplot(4,3,8)
%   high alpha amp chunks
    rose((ph_est(e_fir>0.5*max(e_fir))-ph_fir(e_fir>0.5*max(e_fir))),100);
    subplot(4,3,9)
    plot(e_(0.1e4:0.2e4));%/max(e_(3.8e4:4.2e4)));
    hold on
    plot(e_fir(0.1e4:0.2e4));%/max(e_fir(3.8e4:4.2e4)));
    hold off
    axis tight
    
    subplot(4,3,10)
    plot(lags_e_cfir, xcf_e_cfir);
    grid
    title(num2str([max_xcf_e_cfir, lags_e_this(lag_xcf_e_cfir)]))
    axis([-200 200 0 1])
    subplot(4,3,11)
%   high alpha amp chunks
    rose((ph_cfir(e_fir>0.5*max(e_fir)) - ph_fir(e_fir>0.5*max(e_fir))),100);
    
    subplot(4,3,12)
    plot(e_cfir(0.1e4:0.2e4)/max(e_cfir(0.1e4:0.2e4)));
    hold on
    plot(e_fir(0.1e4:0.2e4)/max(e_fir(0.1e4:0.2e4)));
    hold off
    axis tight
       
end;

