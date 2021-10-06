%%% NEURAL NETWORK DESIGN AND PREDICTION

dehy = readmatrix("datasett_len.xlsx");

% Split data into training (80%) and testing (20%)
tr = 1:0.75*length(dehy);
ts = length(tr):length(dehy);

% Assign features and target datainp = dehy(tr,1:3);
inp = dehy(tr,1:3);
targ = dehy(tr,4);

predin = dehy(ts,1:3);
actarg = dehy(ts,4);

% Scaling the input dataset
for i = 1:3
    x2(:,i)=(inp(:,i)-min(inp(:,i)))/(max(inp(:,i))-min(inp(:,i)));
end
x2;

%Transpose input and target data for MATLAB's accessibility
x = x2.';
t = targ.';

%Choose a Training Function
%For a list of all training functions type: help nntrain
%'trainlm' is usually fastest.
%'trainbr' takes longer but may be better for challenging problems. 
%'trainscg' uses less memory. NFTOOL defaults for low memory situations
trainFcn = 'trainlm'; % Levenberg-Marquadt


%% Optimise for the required number of neurons in the hidden layer
for i=1:50
    % defining architecture of the ANN fitnet
    RandStream.setGlobalStream(RandStream('mt19937ar','seed',10)); % to get constant result
    net.divideFcn = 'divideblock'; % Divide targets (randomly) into three sets using blocks of indices
    hiddenLayerSize = [i i+1];
    net = feedforwardnet(hiddenLayerSize,trainFcn);
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 30/100;
    net.divideParam.testRatio = 0/100;
    
    
    %TRAINING PARAMETERS
    net.trainParam.show = 100; %# of epochs in display
    net.trainParam.lr = 0.0001; %learning rate
    net.trainParam.epochs = 500; %max epochs
    net.trainParam.goal = 0.0005^2; %training goal
    net.performFcn = 'mse'; %Name of a network performance function 
    
    % training the ANN
    [net,tr] = train(net,x,t);
    
    p = net(x); %y = output 
    m = gsubtract(t,p) %e = error between output and target

    performance(i) = perform(net,t,p)
    
    % determining the performance of the ANN model
    yTrain = net(x(:,tr.trainInd));
    yTrainTrue = t(tr.trainInd);
    yVal = net(x(:,tr.valInd));
    yValTrue = t(tr.valInd);
    rmse_train(i) = sqrt(mean((yTrain - yTrainTrue).^2)) %RMSE of training set
    rmse_val(i) = sqrt(mean((yVal - yValTrue).^2)) %RMSE of validation set
end


figure(2)
plot(1:50,rmse_train); hold on;
plot(1:50,rmse_val); hold off;
title('Plot of RMSEs vs Number of Neurons for 4 hidden layers');
xlabel('Number of neurons');
ylabel('RMSE');
legend('RMSE Training','RMSE Validation')

%rmse_val
rmse_vals = rmse_val.';
opt_rmseval = [min(rmse_vals(:))]
opt_num = find(ismember(rmse_vals,opt_rmseval,'rows')>0)

%performance
performances = performance.';
opt_performance = [min(performances(:))]
opt_num = find(ismember(performances,opt_performance,'rows')>0)


%%
% Using our optimal number of neuron in the hidden layer 
RandStream.setGlobalStream(RandStream('mt19937ar','seed',100)); % to get constant result
net.divideFcn = 'divideblock'; % Divide targets (randomly) into three sets using blocks of indices
hiddenLayerSize2 = [36 36 36 36];
net = feedforwardnet(hiddenLayerSize2,trainFcn);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 30/100;
net.divideParam.testRatio = 0/100;

net = init(net);

net.trainParam.show = 100; %# of epochs in display
net.trainParam.lr = 0.0001; %learning rate
net.trainParam.epochs = 700; %max epochs
net.trainParam.goal = 0.0005^2; %training goal
net.performFcn = 'mse'; %Name of a network performance function % help nnperformance

%Train the Network
[net,tr] = train(net,x,t);

view(net)

%Test the Network
y = net(x); %y = output 
e = gsubtract(t,y) %e = error between output and target

figure(2)
plot(e,length(t))
perform = perform(net,t,y)


%Regression Plots for All, Training, Validation, and Testing
yT = y(tr.trainInd);
yV = y(tr.valInd);
yTs = y(tr.testInd);
tT = t(tr.trainInd);
tV = t(tr.valInd);
tTs = t(tr.testInd);
plotregression(tT, yT, 'Training', tV, yV, 'Validation', tTs, yTs, 'Testing', t,y, 'All')


% Plots
% Uncomment these lines to enable various plots.
figure, plotperform(tr)
figure, plottrainstate(tr)
figure, plotfit(net,y,t)
figure, ploterrhist(e)

%% VALIDATING OUR MODEL

% Scaling validation dataset
for i = 1:3
    x_pred(:,i)=(predin(:,i)-min(predin(:,i)))/(max(predin(:,i))-min(predin(:,i)));
end

xpred = x_pred.'
actarg = actarg.'
predTarget = sim(net,xpred);
predTarg = predTarget
figure(3)
%plot(predTarg); hold on; plot(actualTarg, 'r');%hold off
plot(predTarg); hold on; 
plot(actarg, 'r');%hold off
title('Plot of Pred vs Actual for 3 hidden layers');
xlabel('Number of neurons');
ylabel('RMSE');
legend('RMSE Training','RMSE Validation')
hold off
% plot(predin(:,3),predTarg)
error = [predTarg - actarg]

figure(4)
plot(error)
