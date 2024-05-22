function results=ocsvm_ts(x_ts,model)
%
% Testing one-class support vector machine
%
% Input of the function
%    x_ts------Testing instances with size [n_ts,n_dim]
%    model-----Trained OCSVM
%
% Output of the function
%    results---Testing results
%
% Usage: results=ocsvm_ts(x_ts,model)
%
% Date: 2019/3/4

if nargin<2
    help ocsvm_ts
end

n_ts=size(x_ts,1);
[~,~,dec_val]=svmpredict(zeros(n_ts,1),x_ts,model);
results.values=dec_val;
results.labels=sign(dec_val);