function model=ocsvm_tr(x_tr,gamma,nu)
%
% Training one-class support vector machine
%
% Input of the function
%    x_tr--------Input vectors of the training set with size [n_tr,n_dim]
%    gamma-------Parameter in the Gaussian kernel function 
%    nu----------Percentage parameter determinates the upper bound on the
%                fraction of training errors and the lower bound of the
%                fraction of support vectors
%
% Output of the function
%    model-------Trained one-class support vector machine
%
% Usage: model=ocsvm_tr(x_tr,gamma,nu)
%
% Date: 2019/03/04

if nargin<3
    nu=0.1;
end
if nargin<2
    gamma=10;
end
if nargin<1
    help ocsvm_tr
end

n_tr=size(x_tr,1);
label_tr=ones(n_tr,1);

% options:
% -s svm_type : set type of SVM (default 0)
% 	0 -- C-SVC
% 	1 -- nu-SVC
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR
% 	4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -r coef0 : set coef0 in kernel function (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% -m cachesize : set cache memory size in MB (default 100)
% -e epsilon : set tolerance of termination criterion (default 0.001)
% -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)

model=svmtrain(label_tr,x_tr,sprintf('-s 2 -t 2 -g %f -n %f',gamma,nu));