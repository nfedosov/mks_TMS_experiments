function [x_,y_,KK] = kalman_filter(z,F,H,Q,R)

    P_ = Q;
    k = 1;
    x_(:,1) = [0;0];
    for k = 2:length(z)
        x_pred(:,k) = F*x_(:,k-1);
        P_pred = F*P_*F' + Q;
        y_pred(:,k) = z(:,k) - H*x_pred(:,k);
        S = H*P_pred*H' + R;
        K = P_pred*H'*inv(S);
        KK(:,k) = K;
        x_(:,k) = x_pred(:,k) + K*y_pred(:,k);
        P_ = (eye(2) - K*H)*P_pred;
        y_(:,k) = z(:,k) - H*x_(:,k);
    end;

end

