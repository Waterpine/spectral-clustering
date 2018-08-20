load('TDT2_data', 'fea', 'gnd');

% YOUR CODE HERE
fea = full(fea);
[N, ~] = size(fea);
%map
gnd(gnd == 4) = 4;
gnd(gnd == 9) = 1;
gnd(gnd == 13) = 2;
gnd(gnd == 21) = 3;
gnd(gnd == 26) = 5;
gnd_value = unique(gnd);
permute = perms(gnd_value);
% make W
options = [];
options.NeighborMode = 'KNN';
options.k = length(gnd_value);
options.WeightMode = 'Binary';
options.t = 1;
W = constructW(fea,options);

%% spectral
avg_acc = 0;
avg_multual_info = 0;
tmp_idx = zeros(N, 1);
iter = 100;
for j = 1: iter
    idx = spectral(W, length(gnd_value));
    acc = 0;
    mutual_info = 0;
    for i = 1: length(permute)
        map_ = permute(i,:);
        tmp_idx(1:N) = map_(idx(1:N));
        tmp = sum(tmp_idx == gnd) / N;
        if tmp > acc
            acc = tmp;
        end
        tmp = MutualInfo(gnd, tmp_idx);
        if tmp > mutual_info
            mutual_info = tmp;
        end
    end
    avg_acc = avg_acc + acc;
    avg_multual_info = avg_multual_info + mutual_info;
    fprintf('%d\n', j);
end
fprintf('accuracy of spectral is %f\n', avg_acc / iter);
fprintf('mutual info of spectral is %f\n', avg_multual_info / iter);
%% kmeans
avg_acc = 0;
avg_multual_info = 0;
tmp_idx = zeros(N, 1);
iter = 10;
for j = 1: iter
    idx = kmeans(fea, length(gnd_value));
    acc = 0;
    mutual_info = 0;
    for i = 1: length(permute)
        map_ = permute(i,:);
        tmp_idx(1:N) = map_(idx(1:N));
        tmp = sum(tmp_idx == gnd) / N;
        if tmp > acc
            acc = tmp;
        end
        tmp = MutualInfo(gnd, tmp_idx);
        if tmp > mutual_info
            mutual_info = tmp;
        end
    end
    avg_acc = avg_acc + acc;
    avg_multual_info = avg_multual_info + mutual_info;
    fprintf('%d\n',j);
end
fprintf('accuracy of kmeans is %f\n', avg_acc / iter);
fprintf('mutual info of kmeans is %f\n', avg_multual_info / iter);