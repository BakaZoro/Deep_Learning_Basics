# DeepLearning

This repository shall contain the different programs that I will have tried over time:

𝟏. 𝐢𝐫𝐢𝐬𝐝𝐚𝐭𝐚_𝐧𝐨𝐫𝐦𝐚𝐥.𝐩𝐲: MLP on IRIS dataset,Z-score Normalisation, 10 fold cross validation, 1 Hot encoding, 3 layer MLP with sigmoid at hidden layer and softmax at output. Weight initialisation for each fold is done 10 times. And since 10 fold cross validation in itself is repeated 100 times, that makes it 1000 times. No regulirasation is done. Finally the model is compiled, tested and trained with 1000 epochs with each weight initialisation. All the output accuracies are written in a file.
  
𝟐. 𝟑𝐝𝟐.𝐩𝐲: A simple python 3 program for plotting points in 3D, which shall be used in later projects too.

𝟑. 𝐒𝐡𝐮𝐭𝐭𝐥𝐞.𝐩𝐲: The 𝐈𝐑𝐈𝐒 𝐝𝐚𝐭𝐚𝐬𝐞𝐭 contains 150 samples having 3 classes. Each class has 50 samples each. So when the model is trained with "𝐬𝐢𝐠𝐦𝐨𝐢𝐝" as the output function, the initial predictions may/may not be that good. However, what is definitely true that with each epoch, the predicted values move nearer to the actual value. So the program 𝐒𝐡𝐮𝐭𝐭𝐥𝐞.𝐩𝐲, trains a 3 layered MLP on the IRIS dataset and accesses the predicted values for each sample during each epoch. Each predicted value is joined to the actual value by a line. So each cluster of values move nearer to the actual value, causing the line between them to shorten thus making it look like a shuttle moving closer. Each graph is saved as an image and then all those images sequences are then used to compile a video, which is basically a video of how the model learns to predict.

𝟒. 𝐕𝐢𝐝𝐞𝐨.𝐩𝐲: A python program that finds a set of images and then formas a video using those image seqeunces.
