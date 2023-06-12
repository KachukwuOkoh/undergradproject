# Machine and Deep Learning Models Development and Predictions for a Chemical Engineering Project 

## Project Topic: Prediction of Moisture ratio of Harvested Yam using Machine Learning 

![Python](https://user-images.githubusercontent.com/67152646/136196370-2e4f88e4-6784-4ffe-8e49-ae94c3ef952d.PNG)![MATLABB](https://user-images.githubusercontent.com/67152646/136196397-b68a3acf-6634-4189-873e-745cc6c8493e.JPG)

    This folder contains implementation codes used for my undergraduate project with Python and MATLAB. 
    The aim was to predict the moisture ratio of harvested yam 
    Dried yam is beneficial in making materials like flakes, flour and chips from yam slices 
    This project helps the local farmer obtain a moisture ratio value needed for further processing.
    This simulation reduces the number of experiments required to get moisture ratio 
    This also reduces inadequacies as well as time consumption in the laboratory.



## Dataset 

First of, the dataset used was from an experiment conducted on yam dehydration which can be accessed here: https://drive.google.com/file/d/1EUijQdymBHPggFsWUmpaVcR6Abla4EtK/view?usp=sharing.
 
The independent variables comprises:
- Temperature of drying operation 
- Size of the yam material 
- Time taken at specific intervals during the drying operation

The dependent variable is the Moisture Ratio 



## Exploratory Data Analysis

The dataset consists of drying temperatures in the range 65⁰C to 95⁰C, sizes of yam material in the range 1.5mm to 4.5mm, drying times in the range 0 to 320 minutes in different step sizes and their respective moisture ratios.

A preview of the dataset is given below:

   ![Dataset](https://user-images.githubusercontent.com/67152646/137717144-e2d4c798-c769-41ae-b8ff-99dae9d64f71.PNG)



On exploring the dataset, the dataset was found to have different properties. 
- Numerical Features 
- Categorical Features 
- Time-Series Features

Models used as according to the type of dataset explored above were:
## 1. Multi-Layer Feedforward Neural Network (*MLFNN*)

   ![image](https://user-images.githubusercontent.com/67152646/137717542-0fcbd030-97b6-4477-a689-26c5655ec4a0.png)


## 2. Extra Trees Regressor Model


## 3. Long-Short Term Memory Neural Network (*LSTM NN*)

   ![image](https://user-images.githubusercontent.com/67152646/137717640-5a36052c-a2d7-4d75-9fba-c7e96e362cf9.png)





## Models Training 

About 83% was used in training the models while the rest (17%) for testing. The sub-dataset used for testing was of 95 degree celcius, 4.5mm yam material size and 0-320 seconds for time.

An optimization technique was implemented to get the suitable model parameters for the models listed above. This was achieved using a for loop to iterate over a number of parameters and checking for the redundancies in models by using performance metrics like RMSEs, R squared, etc. 

## 1. The *MLFNN* was developed using MATLAB. The model parameters checked for were Hidden Layers and Number of Neurons in each hidden layer. The optimization codes can be found in this folder.  The table below gives a summary statisctics on the *for loop* created to iterate over a given number of hidden layers and number of neurons for optimum parameter determination.

Table 4. 1   Optimum neurons and corresponding RMSEs for hidden layers

Hidden Layers	    1           2           3           4            5

Optimum RMSE	  0.0031	  0.0013	  0.0009      0.0008       0.0009

Optimum Neuron      42          43          27          36           41


Optimum parameters were chosen as 4 Hidden Layers, and 36 Number of Neurons.


## MODEL


   ![image](https://user-images.githubusercontent.com/67152646/137717913-2840c594-7d7b-46a1-a8e4-5d9b6ba2cb5c.png)



## Training Plots


   ![image](https://user-images.githubusercontent.com/67152646/137717934-f724de04-3fd2-406d-b207-505c4920ac6d.png)


## Predicted vs Actual Plot

   ![image](https://user-images.githubusercontent.com/67152646/137720097-f52ff1ba-6bc4-4a73-9b81-eb31c2914a1f.png)


## 2. The *Extra Trees Regressor* was developed using Python. An automated machine learning library called pyCaret was used for this implementation. This library develops different types of regressional models and picks the best one using various performance metrics. As earlier mentioned, the features were seen as categorical here and shuffling was set as True to reduce overfitting on training set


## Variable Importance Plot showing categorical features formed


   ![image](https://user-images.githubusercontent.com/67152646/137717979-48ec4cae-228b-4a60-a9d3-79582fdb58b9.png)



## Predicted versus Actual Plot


   ![image](https://user-images.githubusercontent.com/67152646/137718105-b05be4ab-3861-4974-b285-ac1d4dd0ca6d.png)




## 3. The *LSTM NN* was developed using Python. A part of the codes were referenced from online sites like https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/ and https://apmonitor.com/do/index.php/Main/LSTMNetwork. The model parameters checked for were Hidden Layers, Number of neurons in each Hidden Layer and the Number of effective Time Steps required for a sequence/pattern to be developed.

From the iterative conditions considered as 
1.	Number of neurons from 10 to 90 with a step size of 10 while time step from 2 to 10 with a step size of 2 
2.	Number of neurons from 2 to 10 with a step size of 2 while time step from 10 to 100 with a step size of 10

and using the R squared values (R²) on the test dataset as performance metrics, the following results were extracted for analysis and insights. 
## 1.	Effect of Time Steps on the performance metric R2 with increasing hidden layers

   ![image](https://user-images.githubusercontent.com/67152646/137718860-ff66b9d6-98b6-42bb-b048-fe6b25bf9fa3.png)  ![image](https://user-images.githubusercontent.com/67152646/137718880-1dc65f06-f439-448f-a39f-a4a0c565a83d.png)

   ![image](https://user-images.githubusercontent.com/67152646/137718908-b3ddc8a2-062a-4815-a319-e7c25a361e34.png)  ![image](https://user-images.githubusercontent.com/67152646/137718944-77a08bd4-7a9a-4865-bada-6b87f2a84d67.png)
 



## 2.	Effect of Neurons on the performance metric R2 with increasing time steps

   ![image](https://user-images.githubusercontent.com/67152646/137719030-d34dd99d-681c-452c-a20a-61c74de75a82.png) ![image](https://user-images.githubusercontent.com/67152646/137719050-f8855f9e-14cd-4a17-944f-f703cb193ffa.png)

   ![image](https://user-images.githubusercontent.com/67152646/137719071-f4003cb1-4fff-4dac-a861-c51fb5a9de03.png) ![image](https://user-images.githubusercontent.com/67152646/137719094-04874d0b-1086-4efb-9b5e-0bbe6dcd0022.png)

   ![image](https://user-images.githubusercontent.com/67152646/137719121-81e960e6-9974-47fe-9286-0919d284391a.png)






After much iterations, insights from the figures above, sorting to get the optimum parameters of the highest R², the supposedly best model had parameters as given: Time Steps of 2, 70 Neurons and 3 Hidden Layers.

## MODEL


   ![image](https://user-images.githubusercontent.com/67152646/137719272-8266e55a-507f-4902-85d4-598ab88aa684.png)



## Predicted vs Actual Plot


   ![image](https://user-images.githubusercontent.com/67152646/137719424-8b5f1594-5414-4c67-84ad-f746100f129b.png)







## Models Evaluation

The LSTM, MLFNN and Extra Trees models were seen as suitable predictive models showing values close enough to the actual moisture ratio from experimentation and good performance metrics with R squared values of 0.988, 0.979 and 0.999 respectively.

