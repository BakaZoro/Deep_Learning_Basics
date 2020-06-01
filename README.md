# DeepLearning

This repository shall contain the different programs that I will have tried over time:

𝟏. 𝐢𝐫𝐢𝐬𝐝𝐚𝐭𝐚_𝐧𝐨𝐫𝐦𝐚𝐥.𝐩𝐲: MLP on IRIS dataset,Z-score Normalisation, 10 fold cross validation, 1 Hot encoding, 3 layer MLP with sigmoid at hidden layer and softmax at output. Weight initialisation for each fold is done 10 times. And since 10 fold cross validation in itself is repeated 100 times, that makes it 1000 times. No regulirasation is done. Finally the model is compiled, tested and trained with 1000 epochs with each weight initialisation. All the output accuracies are written in a file.

𝟐. 𝐢𝐫𝐢𝐬𝐝𝐚𝐭𝐚_𝐥𝟏.𝐩𝐲: MLP on IRIS dataset,Z-score Normalisation, 10 fold cross validation, 1 Hot encoding, 3 layer MLP with sigmoid at hidden layer and softmax at output. Weight initialisation for each fold is done 10 times. And since 10 fold cross validation in itself is repeated 100 times, that makes it 1000 times. 𝐋𝟏 𝐫𝐞𝐠𝐮𝐥𝐢𝐫𝐚𝐬𝐚𝐭𝐢𝐨𝐧 using a 𝐩𝐞𝐧𝐚𝐥𝐭𝐲(𝐥𝐚𝐦𝐝𝐚 𝐯𝐚𝐥𝐮𝐞) 𝐨𝐟 𝟎.𝟎𝟏 is done. Finally the model is compiled, tested and trained with 1000 epochs with each weight initialisation. All the output accuracies are written in a file.

𝟑. 𝐢𝐫𝐢𝐬𝐝𝐚𝐭𝐚_𝐥𝟐.𝐩𝐲: MLP on IRIS dataset,Z-score Normalisation, 10 fold cross validation, 1 Hot encoding, 3 layer MLP with sigmoid at hidden layer and softmax at output. Weight initialisation for each fold is done 10 times. And since 10 fold cross validation in itself is repeated 100 times, that makes it 1000 times. 𝐋𝟐 𝐫𝐞𝐠𝐮𝐥𝐢𝐫𝐚𝐬𝐚𝐭𝐢𝐨𝐧 using a 𝐩𝐞𝐧𝐚𝐥𝐭𝐲(𝐥𝐚𝐦𝐝𝐚 𝐯𝐚𝐥𝐮𝐞) 𝐨𝐟 𝟎.𝟎𝟏 is done. Finally the model is compiled, tested and trained with 1000 epochs with each weight initialisation. All the output accuracies are written in a file.

𝟒. 𝟑𝐝𝟐.𝐩𝐲: A simple python 3 program for plotting points in 3D, which shall be used in later projects too.

𝟓. 𝐒𝐡𝐮𝐭𝐭𝐥𝐞.𝐩𝐲: The 𝐈𝐑𝐈𝐒 𝐝𝐚𝐭𝐚𝐬𝐞𝐭 contains 150 samples having 3 classes. Each class has 50 samples each. So when the model is trained with "𝐬𝐢𝐠𝐦𝐨𝐢𝐝" as the output function, the initial predictions may/may not be that good. However, what is definitely true that with each epoch, the predicted values move nearer to the actual value. And since the output function used is "𝐬𝐢𝐠𝐦𝐨𝐢𝐝", it will never give exactly 1 or 0. So the program 𝐒𝐡𝐮𝐭𝐭𝐥𝐞.𝐩𝐲, trains a 3 layered MLP on the IRIS dataset and accesses the predicted values for each sample during each epoch. Each predicted value is joined to the actual value by a '𝐠𝐫𝐞𝐞𝐧 𝐥𝐢𝐧𝐞'. So each cluster of values move nearer to the actual value, causing the line between them to shorten thus making it look like a shuttle moving closer. Each graph is saved as an image and then all those images sequences are then used to compile a video, which is basically a video of how the model learns to predict.

𝟔. 𝐕𝐢𝐝𝐞𝐨.𝐩𝐲: A python program that finds a set of images and then formas a video using those image seqeunces.

𝟕. 𝐌𝐍𝐈𝐒𝐓_𝐌𝐋𝐏.𝐩𝐲:MLP on MNIST dataset,Z-score Normalisation, 10 fold cross validation, 1 Hot encoding, 3 layer MLP with sigmoid at hidden layer and softmax at output. Weight initialisation for each fold is done 10 times. And since 10 fold cross validation in itself is repeated 100 times, that makes it 1000 times. No regulirasation is done. Finally the model is compiled, tested and trained with 20 epochs with each weight initialisation. All the output accuracies are written in a file.

𝟖. 𝐕𝐨𝐜𝐚𝐛𝐎𝐟𝐖𝐨𝐫𝐝𝐬.𝐢𝐩𝐲𝐧𝐛: When dealing with texts as features, the texts need to be pre-processed. One such method is the Bag Of Words Method or BOW. In this the main important words are added to a dictionary and then each sentence is converted to a vector depending on the presence of a particular word in the dictionary. If the word is present then it is 1 else 0. This comes under the feature extraction part. The following code does so. This one creates a dictionary of words.

𝟗. 𝐓𝐞𝐱𝐭𝟐𝐍𝐮𝐦.𝐢𝐩𝐲𝐧𝐛: When handling text data for Machine Learning problems, the data needs to be pre-processed. Even though quite a few ML algorithms nowadays can handle textual data without being converted into numerical data nut many that are implemented using the sklearn library cannot. The tree methods present require the textual data to be converted into numerical form. Hence Text2Num contains three such methods, "CountVectorizer, "TF-IDF Vectroizer" and "Hashing Vectorizer" that help in converting text data to machine readable format.

 
