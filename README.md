# Autonomous-Identification-of-fruit-stock-using-DeepLearning
This project aims to automate identifying the fruits available in a personal storage (inside a refrigerator). This is to enable autonomous ordering/ replenishing of the fruits as required for the user.

Identifying/Recognizing a fruit is challenging digital image processing problem due to the ambiguous nature of the colour, shape and texture of a fruit. Discriminating one fruit from another is a challenging task, especially as a fruit changes shape, colour based on its ripening. Additional complexities are due to factor of obfuscation, lighting and shadow.

This project approaches the problem of identifying fruits from camera image by using Deep Learning Techniques. Deep Learning allows multiple layers of artificial neural network to be trained appropriately.

A dataset of images of nineteen types of fruits used to train and create the Deep Learning model. The Deep Learning system were developed on TensorFlow (backend) using KERAS (API) with PyCharm IDE.

This project also compares three approaches to building the Deep Learning system. The first approach uses Convolutional Neural Network, here the convolutional layer of the Neural Network was developed along with pooling layer, dense layers. Next a Transfer Learning approach was done with the VGG16 model was used on top of which four layers were added. The third approach in Fully Convolutional Neural Network. This do not have flatten or dense layer in the architecture.

Result of these three approaches are presented. The initial scope of the project was to identify the fruits and find the number of fruits of each type. Identification of fruits is completed and presented in the report.
