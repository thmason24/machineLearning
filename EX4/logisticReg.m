%logistic regression and newton's method
clear all; close all; clc;
x = load('ex4x.dat');
y = load('ex4y.dat');

%add yintercept
m = length(y); % store the number of training examples
x = [ones(m, 1), x]; % Add a column of ones to x

% find returns the indices of the
% rows meeting the specified condition
pos = find(y == 1); neg = find(y == 0);

% Assume the features are in the 2nd and 3rd
% columns of x
%plot(x(pos, 2), x(pos,3), '+'); hold on
%plot(x(neg, 2), x(neg, 3), 'o')


g = inline('1.0 ./ (1.0 + exp(-z))'); 
% Usage: To find the value of the sigmoid 
% evaluated at 2, call g(2)

%initialize theta
theta=[0 0 0];

for i=1:10
%neuton's method

h=g(x*theta');
J(i)=(1/(2*m)) * (h-y)'*(h-y);

delJ= 1/m * (h-y)'*x;

H=1/m * x'*diag((h.*(1-h)))*x;
theta = theta - delJ*inv(H);
end

%probability of admitted with exm1=20 and exam2=80
xTest=[1 20 80]
1-g(xTest*theta') 

plot(J)

figure
plot(x(pos, 2), x(pos,3), '+'); hold on
plot(x(neg, 2), x(neg, 3), 'o')
z=[10:0.1:65];
plot(z,-(theta(1) + theta(2)*z)/theta(3))