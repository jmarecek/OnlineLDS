load('../data/setting1.mat')
options.mk = 10;
options.init_w = rand([1,options.mk]);
options.t_tick = 1;


options.lrate = 1;
[RMSE_ogd1,w] = arima_ogd(seq_d1,options);

options.lrate=1.75;
options.epsilon=10^-0.5;
[RMSE_ons1,w] = arima_ons(seq_d1,options);

options.lrate = 10^-3;
[RMSE_ogd0,w] = arima_ogd(seq_d0,options);

options.lrate = 10^3;
options.epsilon=10^-5.5;
[RMSE_ons0,w] = arima_ons(seq_d0,options);

options.initlen = options.mk;
[RMSE_yk,w] = arima_yk(seq_d1,options);

