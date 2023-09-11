% ManhNMF: Manhattan NMF (Guan et al. 2013)
% process_video('NMF', 'ManhNMF', 'dataset/demo.avi', 'output/demo_ManhNMF.avi');
load('datasets/reviews.mat');
M = dtm';
rank = 1;
[W,H] = ManhNMF(M,rank);
L = W' * H;
S = M - L;
