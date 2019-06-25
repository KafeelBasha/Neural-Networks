library(keras)
setwd("D:\\ML\\Neural Network\\Data")

train_x=read.csv("fashion_train.csv",header=TRUE)
train_y=read.csv("fashion_train_labels.csv",header=FALSE,skip=1)
test_x=read.csv("fashion_test.csv",header=TRUE)
test_y=read.csv("fashion_test_labels.csv",header=TRUE)
train_y=train_y[,-1]

#Dimensions
dim(train_x)
length(train_y)
dim(test_x)
length(test_y)

head(train_x[,1:7]) 
head(train_y)
unique(train_y)

#Object labels
tarlab=data.frame(labels=c(0:9),Description=c("T-shirt/Top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"))
View(tarlab)

#Plot
digit=matrix(as.numeric(train_x[1,]),nrow=28)
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(digit),col=grey.colors(255))

#Convert target variable as matrix
train_y=to_categorical(train_y,10) #Class vector to binary class matrix
class(train_y)
colnames(train_y)=c(0:9)
head(train_y)

#Create network architecture
model=keras_model_sequential()
model%>%layer_dense(units=24,input_shape=c(784),activation="sigmoid")%>%layer_dense(units=10,activation="softmax")

model%>%compile(loss="categorical_crossentropy",optimizer=optimizer_sgd(),metrics=c("accuracy"))

train_x=as.matrix(train_x)
history=model%>%fit(train_x,train_y,epochs=30,batch_size=1000,validation_split=0.2)

#Prediction and Accuracy
test_x=as.matrix(test_x)
model%>%predict_classes(test_x)->pclass

#Look at the first image
pclass[1]

filter(tarlab,labels==9)

t1digit=matrix(as.numeric(test_x[1,]),nrow=28)
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(t1digit),col=grey.colors(255))

#Look at the 7th image
pclass[7]
filter(tarlab,labels==6)

t2digit=matrix(as.numeric(test_x[7,]),nrow=28)
rotate <- function(x) t(apply(x, 2, rev))
image(rotate(t2digit),col=grey.colors(255))

test_y=to_categorical(test_y$X0,10)
model%>%evaluate(test_x,test_y)


#Check for Overfit and summarise loss
plot(history)
history
