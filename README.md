# Deep Learning & Tree-Based Models for predictions

![Python](https://user-images.githubusercontent.com/67152646/136196370-2e4f88e4-6784-4ffe-8e49-ae94c3ef952d.PNG)![MATLABB](https://user-images.githubusercontent.com/67152646/136196397-b68a3acf-6634-4189-873e-745cc6c8493e.JPG)

    This folder contains implementation codes used for my undergraduate project with Python and MATLAB. 
    The aim was to predict the moisture ratio of harvested yam 
    Dried yam is beneficial in making materials like flakes, flour and chips from yam slices 
    This project helps the local farmer obtain a moisture ratio value needed for further processing.
    This simulation reduces the number of experiments required to get moisture ratio 
    This also reduces inadequacies as well as time consumption in the laboratory.



## Dataset 

First of, the dataset used were from an experiment conducted on yam dehydration which can be accessed here: .
 
The independent variables comprises:
- Temperature of drying operation 
- Size of the yam material 
- Time taken at specific intervals during the drying operation

The dependent variable is the Moisture Ratio 



## Exploratory Data Analysis

On exploring the dataset, the dataset was found to have different properties. 
- Numerical Features
- Categorical Features 
- Time-Series Features

Models used as according to the type of dataset explored above were:
1. Multi-Layer Feedforward Neural Network (*MLFNN*)
2. Extra Trees Regressor Model
3. Long-Short Term Memory Neural Network (*LSTM NN*)



## Models Training 

About 83% was used in training the models while the rest (17%) for testing.

An optimization technique was implemented to get the suitable model parameters for the models listed above. This was achieved using a for loop to iterate over a number of parameters and checking for the redundancies in models by using performance metrics like RMSEs, R squared, etc. 

1. The *MLFNN* was developed using MATLAB. The model parameters checked for were Hidden Layers and Number of Neurons in each hidden layer. The optimization codes can be found in this folder.

2. The *Extra Trees Regressor* was developed using Python. An automated machine learning library called pyCaret was used for this implementation. This library develops different types of regressional models and picks the best one using various performance metrics. 

3. The *LSTM NN* was developed using Python. A part of the codes were referenced from online sites like https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/ and https://apmonitor.com/do/index.php/Main/LSTMNetwork. The model parameters checked for were Hidden Layers, Number of neurons in each Hidden Layer and the Number of effective Time Steps required for a sequence/pattern to be developed.



## Models Evaluation

The LSTM and Extra Trees models were seen as suitable predictive models showing values close enough to the actual moisture ratio from experimentation and good performance metrics with R squared values of 0.9988 and 0.999 respectively.

