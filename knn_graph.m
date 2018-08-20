function W = knn_graph(X, k, threshold)
%KNN_GRAPH Construct W using KNN graph
%   Input: X - data point features, n-by-p maxtirx.
%          k - number of nn.
%          threshold - distance threshold.
%
%   Output:W - adjacency matrix, n-by-n matrix.

% YOUR CODE HERE
[N, ~] = size(X);
D = EuDist2(X, X);
W = zeros(N, N);
for i = 1: N
    D(i, i) = inf;
    dis = D(i,:);
    dis = sort(dis);
    topk = dis(1:k);
    thres = topk(k);
    Wi = zeros(1, N);
    idx1 = D(i,:) <= thres;
    idx0 = D(i,:) >= threshold;
    Wi(idx1) = 1;
    Wi(idx0) = 0;
    W(i,:) = Wi;
end
W = W | W';
end
