setwd("D:\\ML\\Neural Network\\Data")

train_x=read.csv("fashion_train.csv",header=TRUE)
train_y=read.csv("fashion_train_labels.csv",header=FALSE,skip=1)
test_x=read.csv("fashion_test.csv",header=TRUE)
test_y=read.csv("fashion_test_labels.csv",header=TRUE)
train_y=train_y[,-1]
head(train_x[,1:7])
head(train_y)

train_x=train_x/255
test_x=test_x/255
digit=matrix(as.numeric(train_x[1,]),nrow=28)
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(digit),col=grey.colors(255))

tarlab=data.frame(labels=c(0:9),Description=c("T-shirt/Top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"))

#Constructing the model
library(caret)
control=trainControl(method="cv",number=5)
tune=expand.grid(size=c(9,12),dropout=seq(0.1,0.2,by=0.1),batch_size=c(500,600,1000),lr=0.001,decay=0,rho=0.9,activation="softmax")

gridmodel=train(as.matrix(train_x),as.factor(train_y),method="mlpKerasDropout",trControl=control,tuneGrid=tune,epochs=10)

library(keras)
model=unserialize_model(gridmodel$finalModel$object)

#Prediction and Accuracy
test_y=to_categorical(test_y$X0,10)
model%>%evaluate(as.matrix(test_x),test_y)
