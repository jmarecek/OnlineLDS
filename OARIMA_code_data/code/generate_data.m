%% setting 1

alpha = [0.6, -0.5, 0.4, -0.4, 0.3];
beta = [0.3, -0.2];
noise_mean = 0;
noise_std = 0.3;
seq_num = 10000;


last_seq = noise_std*randn(size(alpha));
last_noise = noise_mean+noise_std*randn(size(beta));
for i = 1:seq_num
    noise_term = noise_mean + noise_std*randn(1);
    seq_d1(i) = alpha*last_seq' + beta*last_noise' + noise_term;
    last_seq = [seq_d1(i),last_seq(1:size(last_seq,2)-1)];
    last_noise = [noise_term,last_noise(1:size(last_noise,2)-1),];
end
save('../data/setting1','seq_d1');

%% undiff
seq_d0=0;
for i = 1:size(seq_d1,2)
    seq_d0(i+1) = seq_d1(i) + seq_d0(i);
end
save('../data/setting1','seq_d0','-append');
clear

%% setting 2
alpha = [0.6, -0.5, 0.4, -0.4, 0.3];
beta = [0.3, -0.2];
seq_num = 5000;
last_seq =  unifrnd(-0.5,0.5,size(alpha));
last_noise =  unifrnd(-0.5,0.5,size(beta));
for i = 1:seq_num
    noise_term =  unifrnd(-0.5,0.5);
    seq1(i) = alpha*last_seq' + beta*last_noise' + noise_term;
    last_seq = [seq1(i),last_seq(1:size(last_seq,2)-1)];
    last_noise = [noise_term,last_noise(1:size(last_noise,2)-1),];
end
alpha = [-0.4,-0.5,0.4,0.4,0.1];
beta = [-0.3, 0.2];
seq_num = 5000;
last_seq =  unifrnd(-0.5,0.5,size(alpha));
last_noise =  unifrnd(-0.5,0.5,size(beta));
for i = 1:seq_num
    noise_term =  unifrnd(-0.5,0.5);
    seq2(i) = alpha*last_seq' + beta*last_noise' + noise_term;
    last_seq = [seq2(i),last_seq(1:size(last_seq,2)-1)];
    last_noise = [noise_term,last_noise(1:size(last_noise,2)-1),];
end

seq_d1 = [seq1,seq2];


save('../data/setting2','seq_d1');

seq_d0=0;
for i = 1:size(seq_d1,2)
    seq_d0(i+1) = seq_d1(i) + seq_d0(i);
end
save('../data/setting2','seq_d0','-append');
clear
%% setting 3
alpha = [0.6, -0.5, 0.4, -0.4, 0.3];
beta = [0.3, -0.2];
noise_mean = 0;
noise_std = 0.3;
seq_num = 15000;


last_seq = noise_std*randn(size(alpha));
last_noise = noise_mean+noise_std*randn(size(beta));
for i = 1:seq_num
    noise_term = noise_mean + noise_std*randn(1);
    seq(i) = alpha*last_seq' + beta*last_noise' + noise_term;
    last_seq = [seq(i),last_seq(1:size(last_seq,2)-1)];
    last_noise = [noise_term,last_noise(1:size(last_noise,2)-1),];
end


%% undiff
diff_seq = seq(5000:15000);
seq1=0;
for i = 1:size(diff_seq,2)
    seq1(i+1) = diff_seq(i) + seq1(i);
end
seq1 = seq1(3:10002);


diff_seq = seq1(5000:10000);
seq2 = 0;
for i = 1:size(diff_seq,2)
    seq2(i+1) = diff_seq(i) + seq2(i);
end
seq2 = seq2(3:5002);

seq2 = [seq(1:5000),seq1(1:5000),seq2(1:5000)];


seq1 = [];
for i = 2:size(seq2,2)
    seq1(i-1) = seq2(i) -seq2(i-1);
end

seq = [];
for i = 2:size(seq1,2)
    seq(i-1) = seq1(i) -seq1(i-1);
end
seq_d2 = seq;
seq_d1 = seq1;
seq_d0 = seq2;
save('../data/setting3','seq_d0','seq_d1','seq_d2');

clear

%% setting 4

alpha1 = [-0.4,0.5,0.4,0.4,0.1];
alpha2 = [0.6,-0.4,0.4,-0.5,0.4];
beta = [0.32, -0.2];
noise_mean = 0;
noise_std = 0.3;
seq_num = 10000;


last_seq = noise_std*randn(size(alpha1));%50 + unifrnd(-5,5,size(alpha))%noise_std*randn(size(alpha));
last_noise = noise_mean+noise_std*randn(size(beta));
for i = 1:seq_num
    noise_term = noise_mean + noise_std*randn(1);
    alpha_t = alpha1*i/10000 + alpha2 *(1- i/10000);
    seq(i) = alpha_t*last_seq' + beta*last_noise' + noise_term;
    
    last_seq = [seq(i),last_seq(1:size(last_seq,2)-1)];
    last_noise = [noise_term,last_noise(1:size(last_noise,2)-1),];
end
seq_d1 = seq;
save('../data/setting4','seq_d1');
%% undiff
seq1=0;
for i = 1:size(seq,2)
    seq1(i+1) = seq(i) + seq1(i);
end
seq_d0 = seq1;
save('../data/setting4','seq_d0','-append');

clear