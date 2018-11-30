### Directories Description

* codeTests:
	1. Results for testing ABF.py
	2. w/o ABF, the histogram should be propotional to exp(-U(x)/kbT)
	3. w ABF, the histogram should be (almost) flat 

* early_stage:
	short ABF run 

* trainingSet:
	training data for NN (long ABF run)

* analysis
	hyperparams for NN (learning_rate and regularization factor alpha included)
	loss per step for NN
	force distribution after NN training 
	plots for force distribution

* pklsave
	weights and biases trained from NN
