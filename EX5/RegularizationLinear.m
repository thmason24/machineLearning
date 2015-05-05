%linear regression with regularization
clear all; close all; clc;

%load linear regression data
x = load('ex5linx.dat');
y = load('ex5liny.dat');

%plot(x,y, 'ro')

%initialize theta
theta = [0 0 0 0 0];

%add yintercept
m = length(y); % store the number of training examples
%generate X vector to form hypothesis
X = [ones(m, 1), x, x.^2, x.^3, x.^4, x.^5]; % Add a column of ones to x


%solve using normal equations
%create matrix that is identity except for theta0 vector
regMat=eye(length(x)-1);
regMat(1,1)=0;


lambdas=[0,1,10];
for i=1:length(lambdas)

lambda = lambdas(i);
theta = inv((X'*X+lambda*regMat))*X'*y;
subplot(1,3,i) 
plot(x,y,"ob", 'markersize',5,"markerfacecolor", "red")
hold on;
k=[-1:0.1:1];
theta
NORM=norm(theta)
plot(k,  theta(1) + k*theta(2)+ k.^2*theta(3) + k.^3*theta(4) + k.^4*theta(5) + k.^5*theta(6) ,"r") 
end
