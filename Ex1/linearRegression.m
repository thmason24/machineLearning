clear all; close all; clc;
x = load('ex2x.dat');
y = load('ex2y.dat');
figure % open a new figure window

%plot data
plot(x, y, 'o');
ylabel('Height in meters')
xlabel('Age in years')

%add yintercept
m = length(y); % store the number of training examples
x = [ones(m, 1), x]; % Add a column of ones to x

%implement linear regression
alpha=0.07; % learning rate

%initialize theta
theta=[0 0];

% execute first step of gradient descent
theta = theta - alpha*(1/m)*(x'*(x*theta'-y))'

%execute 1500 iterations
for i=1:1500
theta = theta - alpha*(1/m)*(x'*(x*theta'-y))';
end
theta

hold on % Plot new data without clearing old plot
plot(x(:,2), x*theta', '-') % remember that x is now a matrix with 2 columns
                           % and the second column contains the time info
legend('Training data', 'Linear regression')
hold off

%check with test data of two boys with age 3.5 and 7 years.   1 added for y intercept
xTest=[1 3.5 ;1 7];
xTest*theta'

%plot J values
J_vals = zeros(100, 100);   % initialize Jvals to 100x100 matrix of 0's
theta0_vals = linspace(-3, 3, 100);
theta1_vals = linspace(-1, 1, 100);
for i = 1:length(theta0_vals)
	  for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = 1/(2*m) * sum((x*t-y).^2); %% YOUR CODE HERE %%
    end
end

% Plot the surface plot
% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped

J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1')

%contour plot
figure;
% Plot the cost function with 15 contours spaced logarithmically
% between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 2, 15))
xlabel('\theta_0'); ylabel('\theta_1')
