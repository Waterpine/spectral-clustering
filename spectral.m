function idx = spectral(W, k)
%SPECTRUAL spectral clustering
%   Input:
%     W: Adjacency matrix, N-by-N matrix
%     k: number of clusters
%   Output:
%     idx: data point cluster labels, n-by-1 vector.

% YOUR CODE HERE


% [N, ~] = size(W);
% D = zeros(N);
% for i = 1: N
%    D(i, i) = sum(W(i,:)); 
% end
% L = D - W;
% [eigenvector, ~] = eigs(L\D, k);
% idx = kmeans(eigenvector, k);


degs = sum(W, 2);
D = sparse(1:size(W, 1), 1:size(W, 2), degs);
L = D - W;
[eigenvector, ~] = eigs(L, k, 'SA');
idx = kmeans(eigenvector, k);

end
