# HPCC_GNN_P02
Optimization of ECL Tensors

## Files
1) ### tf08.ecl
   This module is designed for processing images in the context of neural network applications. The module includes a function named "convertImages," which takes a dataset of image records ("Types.ImgRec") along with   parameters such as the target height, width, channel, and transformation mode. The transformation mode determines the image processing method (e.g., crop_fill, fit, fitv, fith). The function utilizes a Python activity ("pyConvertImages") embedded within ECL to perform image processing using TensorFlow and generates ECL tensors as output. The Python activity uses TensorFlow to resize, crop, and adjust the channels of input images based on the specified parameters. The resulting processed images are converted into ECL tensors, and the module provides functionality to distribute and organize these tensors across nodes in a cluster. 


2) ### Types.ecl
   The module encapsulates image record definition, allowing other parts of the program to use and reference the "ImgRec" structure.

3) ### test1.ecl
   The code processes and transform image data within the context of a Graph Neural Network (GNN) framework. It begins by defining a record structure for images and creating a dataset from a flat file. The code then transforms this dataset, assigning sequential IDs to records while preserving other attributes. Subsequently, it invokes the 'convertImages' module from the 'tf08' module, which is designed for image processing. The module takes the transformed dataset as input, specifying target dimensions and a transformation mode. The output is a tensor dataset representing processed images. Overall, the code demonstrates a pipeline for handling image data, from initial record transformation to invoking a specialized module for image processing within the GNN framework.
   
  

## Introduction
Our module aims to streamline the image processing and tensor conversion tasks, thereby
enabling researchers and practitioners to focus on the core aspects of their work, such as model development, training, and evaluation, also it addressed the requirement of the user to alter the size and channels of the sprayed images. The project hosts various image processing functions, such as resizing, cropping, and adjusting channels, to transform a set of  input images into a single  4D tensors .  These tensors are then organized with associated indexes, facilitating efficient training of artificial neural networks. The  main resizing and channel handling is  done by using python embedding and  parallelization and distribution was mainly done using ECL .

The key image transformational functions include:
- **_crop_fill:_**  Resizes and crops an input image to a specified target size.
- **_fit:_**  Resizes and fits an input image to a specified target size.
- **_fitv:_**  Vertically resizes and fits an input image to a specified target size.
- **_fith:_**  Horizontally resizes and fits an input image to a specified target size.

adjust_channels: Adjusts the number of channels in an input image according to the specified target channels, preserving alpha if applicable.




## Objectives
- An Image library for converting raw images into the tensor format commonly used in GNNs.
- Leverages the power of HPCC systems to enable high-performance and scalable image processing for large-scale datasets.
- Parallel processing of the images dataset to obtain the output.
- To accept the user inputs for resizing and also the target channel requirements
- To generate a stack of the resized images tensors  into a single 4D tenor which is easier for training artificial neural networks.




## Workflow
![The flow of the process of ImageCoverter Module.]()

The process begins with spraying the image dataset onto the cluster provided by HPCC systems which are stored as BLOB (Binary Large Object) and then passed to the ImageConverter Module where the conversion to required ECL Tensors with appropriate image transformation operation happens, which later can be used by various machine learning models.


## Input and Output
- BLOB sprayed Image Dataset is passed as the input to the module, Two datasets have been used. The [first dataset](https://www.kaggle.com/datasets/rounakbanik/pokemon)
consists of 809 images all in JPEG and PNG format and to
scale the working of the module [second dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) is used
which contains 2000 JPEG and BMP formatted images.

- The Output aim to generate a stack of the resized images tensors  into a single 4D tenor which is easier for training artificial neural networks.
  







## Results



## Deployment

Assuming that HPCC cluster is up and running in your computer: -

1) Install ML Core as an ECL bundle by running the below in your terminal or command prompt.
 ```
ecl bundle install https://github.com/hpcc-systems/ML_Core.git
```

2) Install Generalised Neural Networks bundle by running the below in your terminal or command prompt.
 ```
ecl bundle install https://github.com/hpcc-systems/GNN.git
 ```

3) To make sure and also install the required python3 dependencies, please run the Setup.ecl file by running the below command.
 ```
ecl run thor Setup.ecl
```

4) Now that the dependencies have been taken care of, we can run test1.ecl on thor.
 ```
ecl run thor test1.ecl
```

