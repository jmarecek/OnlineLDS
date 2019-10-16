function [RMSE,w] = arma_yk(data,options)
initlen = options.initlen;
mk = options.mk;
list = [];
SE = 0;
AE= 0;
count = 0;
for i = initlen+1:1:size(data,2)

    count = count + 1;
    model = ar(data(1:i-1),mk,'yw');
    y = model.a(2:mk+1)*data(i-1:-1:i-mk)';
    diff = y - data(i);
    SE = SE + diff^2;
    AE = AE + abs(diff);
    if mod(i,options.t_tick)==0
      list = [list; sqrt(SE/count)];
    end
end
 w =  model.a(2:mk+1);
 RMSE = list;