# DeepLearning
This contains different scripts I've tried in Deep Learning

This repository shall contain the different programs that i will have tried over time:

1. MLP on IRIS dataset:
  Z-score Normalisation
  10 fold cross validation
  1 Hot encoding
  3 layer MLP with sigmoid at hidden layer and softmax at output
  Weight initialisation for each fold is done 10 times. And since 10 fold cross validation in itself is repeated 100 times, that     makes it 1000 times
  L1 regulirasation is then used
  Finally the model is compiled, tested and trained with 1000 epochs with each weight initialisation. All the output accuracies are written in a file
  
 2. Same as above, except with L2 regularisation.
