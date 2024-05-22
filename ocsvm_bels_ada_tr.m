function model=ocsvm_bels_ada_tr(X_tr,gamma,nu,T,eta,max_iter)
%
% Training bounded exponential loss function based AdaBoost ensemble of OCSVMs
%
% Input of the function
%    X_tr------Training samples
%    gamma-----Width parameter of Gaussian kernel function 
%              exp(-gamma*||x-y||^2)
%    nu--------Trade-off parameter
%    max_iter--Maximum number of iterations
%
% Output of the function
%    model-----Trained AdaBoost based ensemble of OCSVMs
%
% Usage: model=ocsvm_bels_ada_tr(X_tr,gamma,nu,T,eta,max_iter)
%
% Date: 2019/03/06

if nargin<6
    max_iter=20;
end
if nargin<5
    eta=1;
end
if nargin<4
    T=10;
end
if nargin<3
    nu=1;
end
if nargin<2
    gamma=1;
end
if nargin<1
    help ocsvm_ada_tr
end

n_tr=size(X_tr,1);

w=ones(1,n_tr)/n_tr;
alpha_old=ones(1,T)/T;
f_old=zeros(1,n_tr);

for t=1:T
    if t>1
        % Train weak classifier c_k using the data samples according to w
        rand_num=rand(1,n_tr);
        cw=cumsum(w);
        ind=zeros(1,n_tr);
        for i=1:n_tr
            % Find which bin the random number falls into
            loc=max(find(rand_num(i)>cw))+1;
            if isempty(loc)
                ind(i)=1;
            else
                ind(i)=loc;
            end
        end
    else
        ind=1:n_tr;
    end
    ocsvm{t}=ocsvm_tr(X_tr(ind,:),gamma,nu);
    ypred{t}=ocsvm_ts(X_tr,ocsvm{t});
    labels(t,:)=ypred{t}.labels';
    bata=[];
    beta=find(labels(t,:)==-1);
    epsilon(t)=size(beta,2)/n_tr;
    f_old=f_old+alpha_old(t)*labels(t,:);
    if epsilon(t)==0
        break
    end   
    for j=1:max_iter
        s(t)=eta/(1-exp(-eta))*((epsilon(t)-1)*exp(-alpha_old(t)-eta*exp(-alpha_old(t)))+epsilon(t)*exp(alpha_old(t)-eta*exp(alpha_old(t))));
        H(t)=eta/(1-exp(-eta))*((1-epsilon(t))*exp(-alpha_old(t)-eta*exp(-alpha_old(t)))*(1-eta*exp(-alpha_old(t)))+epsilon(t)*exp(alpha_old(t)-eta*exp(alpha_old(t)))*(1-eta*exp(alpha_old(t))));
        alpha(t)=alpha_old(t)-s(t)/H(t);
        alpha_old(t)=alpha(t);
    end
%     w=w.*(exp(-3/2*eta*exp(-f_old)).^(exp(-alpha(t)*labels(t,:))-1).*exp(-alpha(t)*labels(t,:)));
    w=w.*(exp(-3/2*eta*exp(-f_old)).^(exp(-alpha(t)*(xor(labels(t,:),ones(1,n_tr))*2-1))-1).*exp(-alpha(t)*(xor(labels(t,:),ones(1,n_tr))*2-1)));
    w=w/sum(w);
end

% Resutls
model.ocsvm=ocsvm;
model.alpha=alpha;
model.labels=labels;
model.E=epsilon;