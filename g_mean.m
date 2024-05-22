function acc=g_mean(t_label,y_label)
%
% G-mean measure for the performance of one-class classifier
%
% Input of the function
%    t_label----Target labels
%    y_label----Labels obtained by one-class classifier
%
% Output of the function
%    acc--------Accuracy rate obtained by sqrt(acc_T*acc_O)
%
% Usage: acc=g_mean(t_label,y_label)

if nargin<2
    help g_mean
end

ind_T=find(t_label==1);
acc_T=size(find(y_label(ind_T)==1),1)/size(find(t_label==1),1);
ind_O=find(t_label==-1);
acc_O=size(find(y_label(ind_O)==-1),1)/size(find(t_label==-1),1);

acc=sqrt(acc_T*acc_O);