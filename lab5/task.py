import numpy as np
from utils import *

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	# X[:,0] = np.ones(len(Y))
	Y = Y.astype(float)
	newX = np.ones([len(Y),1])
	for i in range(1,len(X[0])):
		col = np.array(X[:,i])
		if(isinstance(col[0], str)):
			labels = np.unique(col)
			t_col = one_hot_encode(col, labels)
		else:
			# t_col = np.zeros()
			# print(col.std() == 0)
			t_col = (col - col.mean())/col.std()
			t_col = np.array(t_col.reshape(len(Y),1))
			# print(t_col)
			
			# print("/////////////" + t_col.shape)
		t_col = t_col.astype(np.float64)
		newX = np.append(newX, t_col, axis = 1)
	# print(newX.shape)
	# print(np.sum(newX**2,0))
	newX = newX.astype(float)
	# s = newX.sum(0)
	# print(s[71])
	return newX, Y

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	# print(W.shape)
	# X = X.astype(float)
	# W = W.astype(float).reshape(len(W),1)
	# Y = Y.astype(float)
	# print(X.shape)
	# print(W.shape)
	# print(np.matmul(X,W).shape)
	# print(Y.shape)
	temp = Y - np.matmul(X, W)
	# print(grad.shape)
	# print(np.matmul(np.transpose(X), grad).reshape(len(W),1).shape)
	grad = -2*np.matmul(np.transpose(X), temp) + 2*_lambda*W
	# print("//////////")
	# print(grad.shape)
	return grad

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	W = np.zeros([X.shape[1],1])
	for i in range(max_iter):
		# print(i)
		grad = grad_ridge(W, X, Y, _lambda)
		W = W - lr*grad
		# print(W.shape)
		if(np.linalg.norm(grad, ord=2) < epsilon):
			return W
	return W

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	# X = X.astype(float)
	# Y = Y.astype(float)
	# lambdas = lambdas.astype(float)
	n = X.shape[0]
	
	m = int(n/k)
	# print(n,m,k)
	lsse = np.zeros([len(lambdas)])
	for j in range(len(lambdas)):
		ss = 0
		c = 0
		for i in range(k):
			ind1 = i*(m)	
			ind2 = min((i+1)*m , n)
			val_data = X[ind1:ind2][:]
			val_labels = Y[ind1:ind2][:]
			train_data = np.append(X[:ind1][:], X[ind2:n][:], axis=0)
			train_labels = np.append(Y[:ind1][:], Y[ind2:n][:], axis=0)
			W = algo(train_data, train_labels, lambdas[j])
			ss += sse(val_data, val_labels, W)
			c = c+1
		lsse[j] = ss/c 
	return lsse

def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''
	N = X.shape[0]
	D = X.shape[1]
	W = np.zeros([D,1])
	mulxy = X.T @ Y 
	mulxw = X @ W
	for i in range(max_iter):
		for j in range(D):
			col = X[:,j].reshape(N,1)
			# print(col*W[j][0])
			mulxw = mulxw - col*W[j][0]
			den = np.sum(col**2)
			temp = (col.T @ (Y - mulxw))
			if(temp - (_lambda/2) > 0):
				W[j][0] = (temp - (_lambda/2))/den
			elif(temp + (_lambda/2) < 0):
				W[j][0] = (temp + (_lambda/2))/den
			else:
				W[j][0] = 0
			mulxw = mulxw + col*W[j][0]
	return W


if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	lambda1 = 12.4
	lambda2 = 3.4e5
	W1 = ridge_grad_descent(trainX, trainY, lambda1)
	W2 = coord_grad_descent(trainX, trainY, lambda2)
	score1 = sse(testX, testY, W1)
	score2 = sse(testX, testY, W2)
	print(score1)
	print(score2)
	# lambdas = [9.4, 9.8, 10.2, 10.6, 11, 11.4, 11.8, 12.2, 12.4, 12.6, 13, 13.4, 13.8, 14.2, 14.6, 15] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	# lambdas = [11.5, 11.7, 11.9, 12.1, 12.3, 12.5, 12.7, 12.9, 13.1]
	# lambdas = [11, 12]
	# lambdas = [2.4e5, 2.6e5, 2.8e5, 3e5, 3.2e5, 3.4e5, 3.6e5, 3.8e5, 4.0e5, 4.2e5, 4.4e5, 4.6e5, 4.8e5, 5e5, 5.2e5]
	# scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, ridge_grad_descent)
	# plot_kfold(lambdas, scores)