# HPCC_GNN_P02
Optimization of ECL Tensors


- file "tf08.ecl" contains the code for convertImage module
- file "Types.ecl" contains the Image Record structure
- file "test1.ecl" is test file, where call to tf08 module and Types is declared
  

### Introduction




### Objectives




### Workflow



### Input and Output






### Results



### Deployment

Assuming that HPCC cluster is up and running in your computer: -

- Install ML Core as an ECL bundle by running the below in your terminal or command prompt.
 ```ecl bundle install https://github.com/hpcc-systems/ML_Core.git```

- Install Generalised Neural Networks bundle by running the below in your terminal or command prompt.
 ```ecl bundle install https://github.com/hpcc-systems/GNN.git```

- To make sure and also install the required python3 dependencies, please run the Setup.ecl file by running the below command.
 ```ecl run thor Setup.ecl```

- Now that the dependencies have been taken care of,
 ```ecl run thor <filename>```
This should enable you to use the GAN train function given the dataset appropriately.
