%multi-variate linear regression


clear all; close all; clc;
x = load('ex3x.dat');
y = load('ex3y.dat');

%add yintercept
m = length(y); % store the number of training examples
x = [ones(m, 1), x]; % Add a column of ones to x

% scale each feature by it's standard deviation and also subtract the mean
sigma = std(x)
mu=mean(x)
x(:,2) = (x(:,2) - mu(2))./ sigma(2);
x(:,3) = (x(:,3) - mu(3))./ sigma(3);

%initialize theta
theta=[0 0 0];

%find a learning rate alpha.
alpha=0.07; % learning rate

%cost function can be written in the vectorized form as follows
%J=1/(2*m) * (x*theta'-y)' * (x*theta'-y); 


%execute 50 iterations of gradient descent while storing the cost function J at each iteration

for j=1:9
alpha=0.1*3^(j-1);
%initialize theta
theta=[0 0 0];
for i=1:50
J(i)=1/(2*m) * (x*theta'-y)' * (x*theta'-y);
theta = theta - alpha*(1/m)*(x'*(x*theta'-y))';
end
if 1==2
subplot(3,3,j) 
plot(J(1:50), '-')
hold on;
end
end 


%best alpha is 0.9 reinialize with this
alpha = 0.9
%initialize theta
%convergence study
for k=[5 10 50 200 1000 10000]
theta=[0 0 0];
for i=1:k
J(i)=1/(2*m) * (x*theta'-y)' * (x*theta'-y);
theta = theta - alpha*(1/m)*(x'*(x*theta'-y))';
end
k;
theta;
end

k
theta

%predict house value
%remember to scale the feature
xTest = [1 (1650-mu(2))./sigma(2) (3-mu(3))./sigma(3)]
prediction = xTest*theta'

% now calculate using Normal Equation without feature scaling

x = load('ex3x.dat');
y = load('ex3y.dat');

%add yintercept
m = length(y); % store the number of training examples
x = [ones(m, 1), x]; % Add a column of ones to x

theta = (inv(x'*x)*x'*y)'
xTest = [1 1650 3]
prediction = xTest*theta'

%looks good compared to solution!!



