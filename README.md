# HPCC_GNN_P02
Optimization of ECL Tensors


- file "tf08.ecl" contains the code for convertImage module
- file "Types.ecl" contains the Image Record structure
- file "test1.ecl" is test file, where call to tf08 module and Types is declared
  

### Introduction




### Objectives




### Workflow
![The flow of the process of ImageCoverter Module.]()

The process begins with spraying the image dataset onto the cluster provided by HPCC systems which are stored as BLOB (Binary Large Object) and then passed to the ImageConverter Module where the conversion to required ECL Tensors with appropriate image transformation operation happens, which later can be used by various machine learning models.


### Input and Output






### Results



### Deployment

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

