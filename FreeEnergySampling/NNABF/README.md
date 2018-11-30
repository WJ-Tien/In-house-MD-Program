### Directories Description

* codeTests:
	1. Results for testing ABF.py
	2. woABF, the histogram should be propotional to exp(-U(x)/kbT)
	3. wABF, the histogram should be (almost) flat 

* early_stage:
	short ABF run 

* trainingSet:
	training data for NN (long ABF run)

* analysis
	1. hyperparams for NN (learning_rate and regularization factor alpha included)
	2. loss per step for NN
	3. force distribution after NN training 
	4. plots for force distribution

* pklsave
	weights and biases trained from NN
