The ground truth for testing purpose after training.

The testing set in 2 scenarios: Scatter object is 1 (S1) and Scatter objects are 5(S5).

The suffixes: 
 - Positions: X and Y coordinate
 - _true_classes: the true class(es) of the corresponding positions.

Ex: S1_positions(1) is coressponding to S1_true_classes(1,:)

Criteria for testing accuracy:
The neural network predict only a single class for a given position. (one-hot vector).
if the predicted class for a given position is one of the true classes given in the ground-truth, then it is correctly predicted, otherwise wrongly predicted.
