library(keras)
setwd("D:\\ML\\Image processing\\sample")
#Importing images
image1=image_load("Cat.jpg",grayscale=TRUE) #default RGB(3 channels)
image1
image2=image_load("Dog.jpg",grayscale=TRUE) #default RGB(3 channels)
image2

#Images as array of pixel intensities
image1_array=image_to_array(image1)
dim(image1_array)
image2_array=image_to_array(image2)
dim(image2_array)

#Resize images and get pixel intensities
image1_resize=image_load("Cat.jpg",grayscale=TRUE,target_size = c(100,110))
resize_array1=image_to_array(image1_resize)
dim(resize_array1)
image2_resize=image_load("Dog.jpg",grayscale=TRUE,target_size = c(100,110))
resize_array2=image_to_array(image2_resize)
dim(resize_array2)

#Import multiple images and create a matrix of pixel intensities
path="D:\\ML\\Image processing\\sample"
images=list.files(path,pattern=".jpg")
head(images)
length(images)

#Define function to import, resize and extract pixel intensties of images
readimg<-function(files)
{
  pixels<-image_to_array(image_load(files,grayscale=TRUE,target_size =c(100,110))) #default RGB 
}
data=as.data.frame(sapply(images,readimg))
class(data)
dim(data)
#sapply() arranges values in column wise
data=t(data)
dim(data)

#Import images from sub-folders
path="D:\\ML\\Image processing\\Cats and Dogs\\"
pets=list.files(path,recursive=TRUE,pattern=".jpg",full.names=TRUE)
head(pets)
length(pets)

pets=as.data.frame(sapply(pets,readimg))
class(pets)
dim(pets)
#sapply() arranges values in column wise
pets=t(pets)
dim(pets)
head(pets)[,1:4]

#Column of class labels
labels=sapply(strsplit(rownames(pets),"/"),"[",2)
rownames(pets)<-NULL
pets_pv=cbind(pets,class=unlist(labels))
dim(pets_pv)
class(pets_pv)
pets_pv=data.frame(pets_pv)
unique(pets_pv$class)
