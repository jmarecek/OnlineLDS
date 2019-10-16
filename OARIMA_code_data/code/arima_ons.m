function [RMSE,w] = arima_ons(data,options)
mk = options.mk;
lrate = options.lrate;
w = options.init_w;
epsilon = options.epsilon;

list = [];
SE = 0;
A_trans = eye(mk)*epsilon;
for i = mk+1:size(data,2)
    
    diff = w*data(i-mk:i-1)'-data(i);
    grad = 2*data(i-mk:i-1)*diff;

    A_trans = A_trans - A_trans * grad' * grad * A_trans/(1 + grad * A_trans * grad');
    w = w - lrate * grad * A_trans ;

    SE = SE + diff^2;
    if mod(i,options.t_tick)==0
        list = [list; sqrt(SE/i)];

    end
end

RMSE = list;
