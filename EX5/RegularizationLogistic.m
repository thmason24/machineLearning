%logistic regression with regularization
clear all; close all; clc;

%load logistic regression data
x = load('ex5Logx.dat');
y = load('ex5Logy.dat');

figure

% Find the indices for the 2 classes
pos = find(y); neg = find(y == 0);

plot(x(pos, 1), x(pos, 2), '+')
hold on
plot(x(neg, 1), x(neg, 2), 'o')

%initialize theta
%theta = [0 0 0 0 0];

%generate 28 feature vector for polynomial fit
X = map_feature(x(:,1),x(:,2));
m = length(x);

%solve using newton's method
%create regularization matrix that is identity except for theta0 vector
regMat=eye(size(X,2));
regMat(1,1)=0;

g = inline('1.0 ./ (1.0 + exp(-z))'); 
% Usage: To find the value of the sigmoid 
% evaluated at 2, call g(2)


lambdas=[0,1,10];
for i=1:length(lambdas)
	%neuton's method
	lambda=lambdas(i)
	theta = zeros(1,length(X(1,:)));
	for j=1:15
		h=g(X*theta');

		%add regularization term to cost function
		J(j)=(1/(2*m)) * (h-y)'*(h-y) + (lambda/(2*m))*norm(theta(2:end)).^2;
		J(j);

		delJ= 1/m * (h-y)'*X;
		%add regularization term
		delJ(2:end)=delJ(2:end) + (lambda/m)*theta(2:end);

		H=1/m * X'*diag((h.*(1-h)))*X;
		%add regularization term
		H=H+(lambda/m)*regMat;
		
		theta = theta - delJ*inv(H);
	end
	theta(1:6)'
	NORM = norm(theta)

	%plot boundary
	% Define the ranges of the grid
	u = linspace(-1, 1.5, 200);
	v = linspace(-1, 1.5, 200);

	% Initialize space for the values to be plotted
	z = zeros(length(u), length(v));

	% Evaluate z = theta*x over the grid
	for j = 1:length(u)
		for k = 1:length(v)
			% Notice the order of j, i here!
			z(k,j) = map_feature(u(j), v(k))*theta';
		end
	end

	% Because of the way that contour plotting works
	% in Matlab, we need to transpose z, or
	% else the axis orientation will be flipped!
	z = z';
	% Plot z = 0 by specifying the range [0, 0]
	subplot(1,3,i)
	plot(x(pos, 1), x(pos, 2), '+')
	hold on
	plot(x(neg, 1), x(neg, 2), 'o')
	contour(u,v,z, [0, 0], 'LineWidth', 2)
end

