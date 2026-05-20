function [label] = fuzzy(X_train, Y_train, X_test, K, m)
    N_test = size(X_test,1);
    label = zeros(N_test,1);
    classes = unique(Y_train);

    for i = 1:N_test
        x = X_test(i,:);
        dists = sqrt(sum((X_train - x).^2, 2));
        [sortedDists, idx] = sort(dists);
        idx = idx(1:K);
        neighbors = Y_train(idx);
        u = zeros(length(classes),1);

        for j = 1:length(classes)
            for k = 1:K
                if neighbors(k) == classes(j)
                    u(j) = u(j) + (1 / (sortedDists(k)+1e-6))^(2/(m-1));
                end
            end
        end
        [~, label(i)] = max(u);
    end
end
