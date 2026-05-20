function D = KLD(P, Q, k)
%   Wang, Kulkarni & Verdú, IEEE Trans. Info. Theory, 2009

    if nargin < 3
        k = 1;
    end

    [n1, d1] = size(P);
    [n2, d2] = size(Q);
    
    if d1 ~= d2
        error('not same');
    end

    d = d1; 
    n = n1;
    m = n2;


    [~, distP] = knnsearch(P, P, 'K', k + 1);
    rho = distP(:, end); 


    [~, distQ] = knnsearch(Q, P, 'K', k);
    if k > 1
        nu = distQ(:, end);
    else
        nu = distQ; 
    end


    if any(rho == 0)
        warning(['****']);
    end


    D = log(m / (n - 1)) + (d / n) * sum(log(nu ./ rho));
end
