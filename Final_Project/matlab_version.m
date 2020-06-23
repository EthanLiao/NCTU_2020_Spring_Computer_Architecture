clf;clear all;close all;
tic;
SNR_DB = 20;
sig_1 = [1 -1 1 -1 -1 -1 -1 -1 1];
sig_2 = [1 1 -1 1 -1 -1 -1 1 -1];
trans_sig = [sig_1 ; sig_2];
% 2x2 System
H = [[5 6];[7 8]];
mimo_rcv = H*trans_sig;
% ------------------MMSE Process-----------------

% MMSE Dectector
% calculate Signal to noise power ratio : sigm
sigm = 10^(SNR_DB/10);

W = inv(H'*H + 1/sigm * eye(2))*H';
% signal detect
mmse_rcv_sig = W * mimo_rcv;
mmse_rcv_sig(mmse_rcv_sig>0) = 1;
mmse_rcv_sig(mmse_rcv_sig<0) = -1;
whole_time = toc;
% MMSE
figure();
subplot(2,2,1);stem(sig_1);title("MMSE transmission signal branch 1");
subplot(2,2,2);stem(sig_2);title("MMSE transmission signal branch 2");
subplot(2,2,3);stem(mmse_rcv_sig(1,:));title("MMSE recieved signal branch 1");
subplot(2,2,4);stem(mmse_rcv_sig(2,:));title("MMSE recieved signal branch 2");
