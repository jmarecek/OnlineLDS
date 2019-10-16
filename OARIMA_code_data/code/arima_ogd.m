function [RMSE,w] = arima_ogd(data,options)
mk = options.mk;
lrate = options.lrate;
w = options.init_w;

list = [];
SE = 0;

for i = mk+1:size(data,2)    
    diff = w*data(i-mk:i-1)'-data(i);
    w = w - data(i-mk:i-1)*2*diff/sqrt(i-mk)*lrate;

    SE = SE + diff^2;
    if mod(i,options.t_tick)==0
      list = [list; sqrt(SE/i)];
    end
end

 RMSE = list;
