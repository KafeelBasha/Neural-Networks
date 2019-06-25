setwd("D:\\ML\\Neural Network\\Data")

library(keras)
train_x=read.csv("fashion_train.csv",header=TRUE)
train_y=read.csv("fashion_train_labels.csv",header=FALSE,skip=1)
test_x=read.csv("fashion_test.csv",header=TRUE)
test_y=read.csv("fashion_test_labels.csv",header=TRUE)
train_y=train_y[,-1]
head(train_y)

#Object labels
tarlab=data.frame(labels=c(0:9),Description=c("T-shirt/Top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"))

#Create predictor and targetmatrix
train_x=as.matrix(train_x)/255
test_x=as.matrix(test_x)/255
View(train_x[1:2,20:35])

train_y=to_categorical(train_y,10) #Class vector to binary class matrix
test_y=to_categorical(as.matrix(test_y),10) #Class vector to binary class matrix

colnames(train_y)=c(0:9)
colnames(test_y)=c(0:9)

head(test_y)
#Training and Test as arrays
#dim(train_x)<-c(60000,28,28,1)
#test_x=as.matrix(test_x)/255
#dim(test_x)<-c(10000,28,28,1)

#fill the new dimensions in row-major ordering
library(reticulate)
train_x <- array_reshape(train_x, c(nrow(train_x),28,28,1))
test_x <- array_reshape(test_x, c(nrow(test_x),28,28,1))


#Convolutional neural network architechture
model=keras_model_sequential()

#layer_max_pooling_2d(pool_size=c(2,2))%>%%>%layer_conv_2d(filters=16,kernel_size=c(3,3),padding="valid")

model%>%layer_conv_2d(filters=32,kernel_size=c(3,3),padding="same",input_shape=c(28,28,1))%>%layer_conv_2d(filters=16,kernel_size=c(3,3),padding="valid")%>%layer_max_pooling_2d(pool_size=c(2,2))%>%layer_flatten()%>%layer_dropout(0.2)%>%layer_dense(120,activation="relu")%>%layer_dense(84,activation="relu")%>%layer_dropout(0.2)%>%layer_dense(10,activation="softmax")

#Compile
model%>%compile(loss="categorical_crossentropy",optimizer=optimizer_sgd(),metrics="accuracy")

history=model%>%fit(train_x,train_y,batch_size=500,epochs=10,validation_data=list(test_x,test_y))


#Check for Overfit and summarise loss
plot(history)
history

#Prediction
model%>%predict_classes(test_x)->pclass

#Look at the first image
pclass[1]

library(dplyr)
filter(tarlab,labels==9)

class(test_x)
dim(test_x)
element1=test_x[1,,,]
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(element1),col=grey.colors(255))

#Look at the 7th image
pclass[7]
filter(tarlab,labels==4)

element2=test_x[7,,,]
image(rotate(element2),col=grey.colors(255))
