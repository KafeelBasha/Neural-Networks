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

#Create predictor and targetmatrix
train_x=as.matrix(train_x)
train_y=to_categorical(train_y,10) #Class vector to binary class matrix
class(train_y)
colnames(train_y)=c(0:9)
head(train_y)

FLAGS <- flags(
  flag_numeric("dropout1", 0.2),
  flag_integer("batch_size", 600)
)

#Another way to regularise is to use dropout layer
modeld=keras_model_sequential()
modeld%>%layer_dense(units=40,input_shape=c(784),activation="sigmoid")%>%layer_dropout(rate=FLAGS$dropout1,seed=123)%>%layer_dense(units=10,activation="softmax")

modeld%>%compile(loss="categorical_crossentropy",optimizer=optimizer_sgd(),metrics=c("accuracy"))

dhist=modeld%>%fit(train_x,train_y,epochs=20,batch_size=FLAGS$batch_size,validation_split=0.2)
