# Deep Learning Labs

As computation power increases everyday, models that are more complex and that can process more data become possible. This field, better known as deep learning is
a subset of Machine Learning, the buzz word in the industry and academia alike, with some added benefits. In Machine Learning tasks, the programmer must
generate features from the data before using a mathematical model to fit the data. In deep learning, however, the model itself learns important features from the data.
Along with that, much more complicated functions can be modelled and hence, a myriad of tasks become possible. In this lab, you will learn some of those tasks, ranging from
understanding images to comprehend natural language. You will learn to solve these tasks using two of the most pervasive platforms,namely Keras and Pytorch.

The labs are divided broadly into three parts:
 - Vision - This part involves processing image datasets to make various inferences. Training the model to identify what an image represents, highlighting 
 a particular object in it, identifying  certain objects in an image, transferring art styles from one painting to the other and so on, are examples where images are taken as inputs and models should be able to successfully 'see' them. Labs 1,2,3,9 and 11 deal with Vision tasks. 
 - Natural Language Processing - This part involves understanding the language spoken by humans. We all have been taught that the computer understands only '1's and '0's, but in these tasks, you will see that it is also possible for the computer to understand human language by finding patterns in the dataset and representing them in binary('1's and '0's). Some examples of this section include Language to Language translation, Extracting Answers to Questions in a Language, Identifying emotions from reviews and so on. Labs 4,5,6 and 7 deal with NLP tasks.  
 - Autoencoders and GANs - These are some additional exercises which are a bit different from the much bigger domains of Vision and Natural Language. These are generative models, or models which create images or text, instead of making any decisions based on them. They can be useful for various tasks like anomaly detection, where general trends are created and an alert is issued in case of aberrant behaviour. Labs 8 and 12 deal with this task.
 
 More details about these are as follows:
 
 ## Lab 1 - MNIST Classification
 
 In this lab, you will build a model in Keras for classifying images of handwritten digits(0-9). This task is known as multi-class classification and
 is useful for various applications such as classification of diseases based on X-Rays in Medical Science to classifying bird sounds in zoology.
 You will build a simple Convolutional Network for this purpose.
 
 ## Lab 2 - Image Segmentation
 
 In this lab, you will perform image segmentation on the Pet dataset. Given an image, you will train the model to find regions in the image which are pets. Image segmentation,or classifying each pixel as belonging to an object or not, finds applications from medical science where it is used to detect diseases like brain tumor, to industry, where it is used for Non Destructive Analysis of components.
 
 ## Lab 3 - Object Detection
 
 In this lab, you will perform the task of object detection on the COCO dataset. This problem involves making a boundary box around a specific object, and then identifying what that object is. This task is useful for applications such as tracking. Many industries have recently built systems using Object Detection to detect whether an employee has wore a mask or not, and are maintaining social distancing or not, for tackling the COVID crisis.
 
 ## Lab 4 - Sentiment Classification
 
 In this lab, you will build a model which takes a movie review as input and classifies it as being positive or negative. You will be using the dataset provided by IMDB for this task. Sentiment analysis has gained a lot of interest since it is directly connected with human psychology. For example, platforms like Twitter and Facebook perform this task to filter out any messages which may be offensive.
 
 ## Lab 5 - Using pretrained embeddings for text classification
 
 In this lab, you will again perform the task of text classification, but using pre-trained word embeddings. You will be using GloVe embeddings for classifying News Dataset. Word embeddings are vector representations of words. For example, the vector of 'king' minus the vector of 'queen' plus the vector of 'girl' should be very close to the vector of 'boy'. Such semantic meanings are captured using word embeddings. Here, each of the dimensions in the vector space represent some phsical quantity as being present or absent and to what degree. You will see that using these makes it much easier to perform tasks such as text classification, which require understanding the semantics of the text.
 
 ## Lab 6 - Question Answering
 
 In this lab, you will train a model to extract answers to an input question from an input paragraph. The dataset used for this is the SQuAD dataset or the Stanford Question Answering Dataset. We will build a model known as BERT transformer for this purpose. Question answering is being used to generate medical helper bots for assisting doctors.
 
 ## Lab 7 - Sequence to Sequence Modelling

In this lab, you will carry out addition of two numbers which are represented as strings. Though this is a simple task by converting them to integers, solving it as an NLP problem helps understand the basics of sequence to sequence modelling, where the input is a sequence of text, and the output is also a sequence of text which may be of same or different length. This is used for problems like translating one language to other, summarizing documents and so on.

## Lab 8 - Using Auto Encoders for Anomaly Detection

In this lab, you will work with time series data. The task is to find out whether there is an anomaly seen in the input data. The model is an autoencoder, which is trained on normal data. The autoencoder encodes the data and then decodes it again to form the original data. When it comes across an anomaly during testing, it outputs a huge loss, signalling an anomaly. This is used on a large scale, but not exclusively, for detecting anomalies in the Civil industry and the Electronic Industry.

## Lab 9 - Neural Style Transfer

In this lab, you will build a model to generate one image in the art style of the other. In this example, you will apply the art style of Van Gogh's 'A Starry Night' to a painting of Paris. On a higher level, one model first learns the 'art' features of an image and uses it to reconstruct a base image so that it is similar to the base image but with the style of the other image. This concept is becoming quite popular for 'reviving' deceased personalities by making the model mimic their behaviour.

## Lab 10 - Getting Started with Pytorch

In this lab, you will learn how to generate a simple model in Pytorch.

## Lab 11 - Transfer Learning

In this lab, you will again perform the task of image classification, but with a limited dataset. You will use models which are pre-trained on similar tasks and fine tune them to learn the required data. In this example, you will use Pytorch to fine tune the ResNet to classify between bees and ants. ResNet has been trained for multi-class image classification on the ImageNet dataset but labels like 'ants' or 'bees' were not seen by the model during training. This technique of fine tuning pre-trained models in the absence of a large amount of data is known as transfer learning and is used extensively for getting better and faster results.

## Lab 12 - DCGANs

In this lab, you will build a generative adversarial network(or GAN) known as DCGAN. You will teach the model how to generate faces of celebrities by showing it the faces of real celebrities. Generative networks find applications in tasks like creating cartoons, generating artificial data for training other models, applications like face aging, 3D object generation and so on. As a fun exercise, you can try feeding it some images of yourself and see what the model outputs.

Note that while some examples are done in Keras and some in Pytorch, both are quite powerful platforms. Each of the examples can be done in either of the two. This is an optional part but it is highly encouraged to try out different problems using the different platforms to get a know-how of how they can be used for building and training different kinds of models
