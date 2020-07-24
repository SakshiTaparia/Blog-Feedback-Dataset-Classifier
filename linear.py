import sys
import numpy as np
import scipy
import math

import sklearn
from sklearn import linear_model

if (sys.argv[1] == 'a'):

	# Open file in read text mode
	train_data_raw = open(sys.argv[2], 'rt')

	# Use loadtxt() function from Numpy to import data directly
	train_data = np.loadtxt(train_data_raw, delimiter=",")
	
	# print(train_data.shape)
	# indexing in Python starts from 0

	y = train_data[:, train_data.shape[1]-1]
	
	x = train_data[: , :-1]
	x1 = np.ones((train_data.shape[0],1))
	
	x_f = np.hstack((x1,x))
	# print(x_final)

	x_tf = x_f.transpose()
	w = np.dot(np.dot(np.linalg.inv(np.dot(x_tf,x_f)), x_tf), y)

	orig_stdout = sys.stdout
	
	f = open(sys.argv[5], 'w')
	sys.stdout = f
	
	for i in range (0, train_data.shape[1]):
	    print(w[i])

	test_data_raw = open(sys.argv[3], 'rt')
	test_data = np.loadtxt(test_data_raw, delimiter=",")

	X1 = np.ones((test_data.shape[0],1))
	test_data_final = np.hstack((X1, test_data))

	answer = np.dot(test_data_final, w)
	
	g = open(sys.argv[4], 'w')
	sys.stdout = g
	
	for i in range (0, test_data_final.shape[0]):
	    print(answer[i])
	    print('\n')

elif (sys.argv[1] == 'b'):

	train_data_raw = open(sys.argv[2], 'rt')
	train_data = np.loadtxt(train_data_raw, delimiter=",")
	
	#10-fold cross-validation
	
	fold=10
	
	data = []
	split_size = int(train_data.shape[0]/fold)
	
	for i in range(0, fold):
		
		if(i == fold-1):
			data.append(train_data[i*split_size:max((i+1)*split_size,train_data.shape[0])])
			break
		
		# validation set
		data.append(train_data[i*split_size:(i+1)*split_size])
	
	lambda_data_raw = open(sys.argv[4],'rt')
	lambda_set = np.loadtxt(lambda_data_raw,delimiter='\n')
	
	minerr = 10**20

	for cur_lambda in lambda_set:
		
		# print(cur_lambda, end=':')
		
		avg_error = 0.0

		for k in range(0,fold):
			
			cur_train_data = []
			
			for i in range(0, fold):
				
				if i!=k:
					
					for j in data[i]:
						cur_train_data.append(j)
			
			cur_train_data = np.array(cur_train_data)
			#training data obtained
			
			cur_test_data = data[k]
			cur_test_data = cur_test_data[:,:-1]

			#to be used to calculate error
			y_real = np.transpose(data[k])[-1]

			y = cur_train_data[:, cur_train_data.shape[1]-1]
	
			x = cur_train_data[: , :-1]
			x1 = np.ones((cur_train_data.shape[0],1))
			
			x_f = np.hstack((x1,x))
			# print(x_final)

			x_tf = x_f.transpose()
			w = np.dot(np.dot(np.linalg.inv(np.dot(x_tf,x_f) + cur_lambda*(np.eye(cur_train_data.shape[1]))), x_tf), y)
 			
			X1 = np.ones((cur_test_data.shape[0],1))
			cur_test_data_final = np.hstack((X1, cur_test_data))

			answer = np.dot(cur_test_data_final, w)
			
			# print(answer.shape)
			# g = open(sys.argv[4], 'w')
			# sys.stdout = g
			
			error = np.dot(answer-y_real,answer-y_real);
			
			avg_error = avg_error + error
		
		avg_error = avg_error/fold
		
		if avg_error < minerr:
			minerr = avg_error
			min_lamb = cur_lambda

		# print(avg_error)

	print(min_lamb)

	y = train_data[:, train_data.shape[1]-1]
	
	x = train_data[: , :-1]
	x1 = np.ones((train_data.shape[0],1))
	x_f = np.hstack((x1,x))

	x_tf = x_f.transpose()
	lmbda = min_lamb
	w = np.dot(np.dot(np.linalg.inv(np.dot(x_tf,x_f) + lmbda*(np.eye(train_data.shape[1]))), x_tf), y)

	orig_stdout = sys.stdout

	f = open(sys.argv[6], 'w')
	sys.stdout = f
	
	for i in range (0, train_data.shape[1]):
	    print(w[i])

	test_data_raw = open(sys.argv[3], 'rt')
	test_data = np.loadtxt(test_data_raw, delimiter=",")

	X1 = np.ones((test_data.shape[0],1))
	test_data_final = np.hstack((X1, test_data))

	answer = np.dot(test_data_final, w)
	
	g = open(sys.argv[5], 'w')
	sys.stdout = g
	
	for i in range (0, test_data_final.shape[0]):
	    print(answer[i])
	    
elif (sys.argv[1] == 'c'):
	
	train_data_raw = open(sys.argv[2], 'rt')
	train_data = np.loadtxt(train_data_raw, delimiter=",")
	
	n = train_data.shape[1];

	test_data_raw = open(sys.argv[3], 'rt')
	test_data = np.loadtxt(test_data_raw, delimiter=",") 
	
		
	# for i in range (0, n):
	#     train_data = np.hstack((train_data, np.tanh(train_data[:, [i]])))
	#     train_data = np.hstack((train_data, np.maximum(train_data[:, [i]],0)))
	#     test_data = np.hstack((test_data, np.tanh(test_data[:, [i]])))
	#     test_data = np.hstack((test_data, np.maximum(test_data[:, [i]], 0)))
	    

	fold = 10
	
	data = []
	split_size = int(train_data.shape[0]/fold)
	
	for i in range(0, fold):
		
		if(i == fold-1):
			data.append(train_data[i*split_size:max((i+1)*split_size,train_data.shape[0])])
			break
		
		# validation set
		data.append(train_data[i*split_size:(i+1)*split_size])
	
	lambda_set =  [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
	
	minerr = 10**20

	for cur_lambda in lambda_set:
		
		avg_error = 0.0

		for k in range(0,fold):
			
			cur_train_data = []
			
			for i in range(0,fold):
				
				if i!=k:
					
					for j in data[i]:
						cur_train_data.append(j)
			
			cur_train_data = np.array(cur_train_data)
			#training data obtained
			
			cur_test_data = data[k]
			cur_test_data = cur_test_data[:,:-1]

			#to be used to calculate error
			y_real = np.transpose(data[k])[-1]

			y = cur_train_data[:, cur_train_data.shape[1]-1]
	
			x = cur_train_data[: , :-1]
			x1 = np.ones((cur_train_data.shape[0],1))
			
			x_f = np.hstack((x1,x))
			

			w = linear_model.LassoLars(cur_lambda, fit_intercept = False) 
			w.fit(x_f,y)

			X1 = np.ones((cur_test_data.shape[0],1))
			cur_test_data_final = np.hstack((X1, cur_test_data))

			answer = np.dot(cur_test_data_final, w.coef_)

			error = np.dot(answer-y_real,answer-y_real);

			avg_error = avg_error + error
		
		avg_error = avg_error/fold
		
		if avg_error < minerr:
			minerr = avg_error
			min_lamb = cur_lambda

	print(min_lamb)

	y = train_data[:, train_data.shape[1]-1]
	
	x = train_data[: , :-1]
	x1 = np.ones((train_data.shape[0],1))
	
	x_f = np.hstack((x1,x))

	w = linear_model.LassoLars(min_lamb, fit_intercept = False)
	w.fit(x_f,y)

	w_arr = np.array(w.coef_)

	print(w_arr)
	
	# orig_stdout = sys.stdout

	# X1 = np.ones((test_data.shape[0],1))
	# test_data_final = np.hstack((X1, test_data))
	# answer = np.dot(test_data_final, w_arr)

	# g = open(sys.argv[4], 'w')
	# sys.stdout = g
	
	# for i in range (0, test_data_final.shape[0]):
	# 	print(answer[i])


	## CHANGING THE TRAINING DATA BASED ON LASSO ##

	train_data_new = np.ones((train_data.shape[0],1))
	test_data_new = np.ones((test_data.shape[0],1))

	for i in range (1, w_arr.shape[0]):

		if (abs(w_arr[i]) > 0.00000001):
			
			train_data_new = np.hstack((train_data_new, train_data[:, [i-1]]))
			test_data_new = np.hstack((test_data_new, test_data[:, [i-1]]))
						
	n = train_data_new.shape[1]

	for i in range (1, n):

		#train_data_new = np.hstack((train_data_new, np.reciprocal(np.abs(train_data[:, [i-1]])+1) ))
		#train_data_new = np.hstack((train_data_new, np.log(np.abs(train_data[:, [i]])+1) ))
		#train_data_new = np.hstack((train_data_new, np.abs(train_data[:, [i-1]])))
		#train_data_new = np.hstack((train_data_new, np.power(train_data[:, [i-1]],2)))
		#train_data_new = np.hstack((train_data_new, np.power(train_data[:, [i-1]],3)))
		#train_data_new = np.hstack((train_data_new, np.exp(np.multiply(train_data[:, [i]], -1)) ))
		#train_data_new = np.hstack((train_data_new, np.reciprocal(np.exp(np.multiply(train_data[:, [i-1]], -1))+1)))
		train_data_new = np.hstack((train_data_new, np.tanh(train_data[:, [i-1]])))
		train_data_new = np.hstack((train_data_new, np.maximum(train_data[:, [i-1]], 0)))
		
		#test_data_new = np.hstack((test_data_new, np.reciprocal(np.abs(test_data[:, [i-1]])+1) ))
		#test_data_new = np.hstack((test_data_new, np.log(np.abs(test_data[:, [i]])+1) ))
		#test_data_new = np.hstack((test_data_new, np.abs(test_data[:, [i-1]])))
		#test_data_new = np.hstack((test_data_new, np.power(test_data[:, [i-1]],2)))
		#test_data_new = np.hstack((test_data_new, np.power(test_data[:, [i-1]],3)))
		#test_data_new = np.hstack((test_data_new, np.exp(np.multiply(test_data[:, [i]], -1)) ))
		#test_data_new = np.hstack((test_data_new, np.reciprocal(np.exp(np.multiply(test_data[:, [i-1]], -1))+1)))
		test_data_new = np.hstack((test_data_new, np.tanh(test_data[:, [i-1]])))
		test_data_new = np.hstack((test_data_new, np.maximum(test_data[:, [i-1]], 0)))
		
	for i in range (1, n-1):
	    
	    for j in range (i+1, n):
	    
	        train_data_new = np.hstack((train_data_new, np.multiply(train_data[:, [i]], train_data[:, [j]]) ))
	        #train_data_new = np.hstack((train_data_new, np.abs(np.multiply(train_data[:, [i]], train_data[:, [j]])) ))
	        test_data_new = np.hstack((test_data_new, np.multiply(test_data[:, [i]], test_data[:, [j]]) ))
	        #test_data_new = np.hstack((test_data_new, np.abs(np.multiply(test_data[:, [i]], test_data[:, [j]])) ))
		    
    # print(train_data_new.shape)
	# print(train_data_new)

	train_data_new = np.hstack((train_data_new, train_data[:, [train_data.shape[1]-1]]))

	###########################################################################################

	fold=10
	
	data = []
	split_size = int(train_data_new.shape[0]/fold)
	
	for i in range(0, fold):
		
		if(i == fold-1):
			data.append(train_data_new[i*split_size:max((i+1)*split_size,train_data_new.shape[0])])
			break
		
		# validation set
		data.append(train_data_new[i*split_size:(i+1)*split_size])
	
	lambda_set =  [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3,0, 10.0, 30.0, 100.0, 300.0]
	
	minerr = 10**20

	for cur_lambda in lambda_set:
		
		avg_error = 0.0

		for k in range(0,fold):
			
			cur_train_data = []
			
			for i in range(0,fold):
				
				if i!=k:
					
					for j in data[i]:
						cur_train_data.append(j)
			
			cur_train_data = np.array(cur_train_data)
			#training data obtained
			
			cur_test_data = data[k]
			cur_test_data = cur_test_data[:,:-1]

			#to be used to calculate error
			y_real = np.transpose(data[k])[-1]

			y = cur_train_data[:, cur_train_data.shape[1]-1]
	
			x = cur_train_data[: , :-1]

			w = linear_model.LassoLars(cur_lambda, fit_intercept = False) 
			w.fit(x,y)

			answer = np.dot(cur_test_data, w.coef_)
			
			error = np.dot(answer-y_real,answer-y_real);

			avg_error = avg_error + error
		
		avg_error = avg_error/fold
		
		if avg_error < minerr:
			minerr = avg_error
			min_lamb = cur_lambda

	print(min_lamb)

	y = train_data_new[:, train_data_new.shape[1]-1]
	x = train_data_new[: , :-1]

	w = linear_model.LassoLars(min_lamb, fit_intercept = False)
	w.fit(x,y)

	w_arr = np.array(w.coef_)

	####################################################################

	orig_stdout = sys.stdout

	answer = np.dot(test_data_new, w_arr)
	
	g = open(sys.argv[4], 'w')
	sys.stdout = g
	
	for i in range (0, test_data_new.shape[0]):
		print(answer[i])

else:
	print("Invalid input!") 
