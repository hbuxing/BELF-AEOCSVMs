clear all
close all

% Generate the noise-free samples
n_normal=200;
x1=0.4+0.2*rand(n_normal/4,1);
y1=0.4+2.2*rand(n_normal/4,1);
x2=2.4+0.2*rand(n_normal/4,1);
y2=0.4+2.2*rand(n_normal/4,1);
x3=0.4+2.2*rand(n_normal/4,1);
y3=0.4+0.2*rand(n_normal/4,1);
x4=0.4+2.2*rand(n_normal/4,1);
y4=2.4+0.2*rand(n_normal/4,1);
XT=[[x1;x2;x3;x4],[y1;y2;y3;y4]];

% Generate noise
n_novel=50;
x5=3*rand(n_novel,1);
y5=3*rand(n_novel,1);
XO=[x5,y5];

X=[XT;XO];
n_T=size(XT,1);
n_O=size(XO,1);

labels=[ones(n_T,1);-ones(n_O,1)];

% Scale the training samples into [-1,1]
n_data=size(X,1);
X_scale=(X-ones(n_data,1)*min(X))./(ones(n_data,1)*(max(X)-min(X)))-2;

% Generate the grid data to demonstrate the results
x_min_gd=min(X(:,1))-0.05;
x_max_gd=max(X(:,1))+0.05;
y_min_gd=min(X(:,2))-0.05;
y_max_gd=max(X(:,2))+0.05;
dx=100;
dy=100;
[x_grid,y_grid]=meshgrid(x_min_gd:(x_max_gd-x_min_gd)/(dx-1):x_max_gd,y_min_gd:(y_max_gd-y_min_gd)/(dy-1):y_max_gd);
X_grid=[x_grid(:) y_grid(:)];
% Scale the grid data into [-1,1]
n_grid=size(X_grid,1);
X_grid_scale=(X_grid-ones(n_grid,1)*min(X))./(ones(n_grid,1)*(max(X)-min(X)))-2;

% Plot the training samples
plot(XT(:,1),XT(:,2),'b.','markersize',10)
hold on
plot(XO(:,1),XO(:,2),'rx','markersize',10,'linewidth',2)
title('\it Square-Outlier')

% OCSVM
gamma=2^5;
nu=0.2;
ocsvm=ocsvm_tr(X_scale,gamma,nu);
ypred=ocsvm_ts(X_scale,ocsvm);
ypred_grid=ocsvm_ts(X_grid_scale,ocsvm);
y_val=ypred_grid.values;
labels_ocsvm=ypred.labels;
g_ocsvm=g_mean(labels,labels_ocsvm);
figure(2)
plot(X(labels_ocsvm==1,1),X(find(labels_ocsvm==1),2),'b.','markersize',10);
hold on
plot(X(find(labels_ocsvm==-1),1),X(find(labels_ocsvm==-1),2),'rx','markersize',10,'linewidth',2);
contour(x_grid,y_grid,reshape(y_val,size(x_grid)),[0,0],'k-','linewidth',2);
title('OCSVM')

% Bounded exponential loss function based AdaBoost ensemble of OCSVMs
T=20;
eta=0.1;
max_iter=20;
ocsvm_bels_ada=ocsvm_bels_ada_tr(X_scale,gamma,nu,T,eta,max_iter);
ypred_ocsvm_bels_ada=ocsvm_bels_ada_ts(X_scale,ocsvm_bels_ada);
ypred_grid_ocsvm_bels_ada=ocsvm_bels_ada_ts(X_grid_scale,ocsvm_bels_ada);
y_val_ocsvm_bels_ada=ypred_grid_ocsvm_bels_ada.values;
labels_ocsvm_bels_ada=sign(ypred_ocsvm_bels_ada.values);
g_ocsvm_bels_ada=g_mean(labels,labels_ocsvm_bels_ada);
figure(3)
plot(X(find(labels_ocsvm_bels_ada==1),1),X(find(labels_ocsvm_bels_ada==1),2),'b.','markersize',10);
hold on
plot(X(find(labels_ocsvm_bels_ada==-1),1),X(find(labels_ocsvm_bels_ada==-1),2),'rx','markersize',10,'linewidth',2);
contour(x_grid,y_grid,reshape(y_val_ocsvm_bels_ada,size(x_grid)),[0,0],'k-','linewidth',2);
title('BELF-AEOCSVMs')