function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;
F1_vec = [];
prec_vec = [];
rec_vec = [];
stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    predictions = pval < epsilon;
	%number of false positives
	fp = sum(predictions == 1 & yval == 0);
	%number of false negatives
	fn = sum(predictions == 0 & yval == 1);
	%number of true positives
	tp =  sum(predictions == 1 & yval == 1);
	%calculate precision
	prec = tp /(tp + fp);
	%calculate recall
	rec  = tp / (tp + fn);
	%calculate f1 score
	F1 = (2 * prec * rec ) / (prec + rec);
	F1_vec = [F1_vec F1];
	prec_vec = [prec_vec prec];
	rec_vec = [rec_vec rec];
	

    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end
%figure
%plot( min(pval):stepsize:max(pval), F1_vec, min(pval):stepsize:max(pval), rec_vec , min(pval):stepsize:max(pval), prec_vec )

end
