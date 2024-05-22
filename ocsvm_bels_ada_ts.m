function ypred=ocsvm_bels_ada_ts(X_ts,ocsvm_bels_ada)
%
% Prediction output of the trained bounded exponential loss function bsed 
% AdaBoost ensemble of OCSVMs
%
% Input of the function
%    X_ts------------Test samples
%    ocsvm_bels_ada--The trained AdaBoost based ensemble of OCSVMs
%
% Output of the function
%    ypred-------Function value and sign values of the decision function 
%                upon the test set
%
% Usage: ypred=ocsvm_bels_ada_ts(X_ts,ocsvm_bels_ada)
%
% Date: 2019/03/07

if nargin<2
    help ocsvm_bels_ada_ts
end

ocsvm=ocsvm_bels_ada.ocsvm;
alpha=ocsvm_bels_ada.alpha;
T=length(ocsvm);
n_ts=size(X_ts,1);
for t=1:T  
     ypreds{t}=ocsvm_ts(X_ts,ocsvm{t});
end

tmp=zeros(1,n_ts);
for t=1:T
    tmp=tmp+alpha(t)*ypreds{t}.labels';
end
ypred.values=tmp';   
ypred.labels=sign(tmp)';