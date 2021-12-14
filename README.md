# hcunit
### Introduction
Homomorphic Convolution Unit (hcunit), including homomorphic convolution layer, homomorphic average pooling layer, square activation layer and fully connected layer, uses the leveled homomorphic encryption (CKKS) to encrypt the data inputted into model.
The whole computations in model are in ciphertext forms. The cloud service providers and model providers can not get any useful information about the input and the results. 
We will get the encrypted inference result and decrypt it to get the real result.
By using this homomorphic convolution unit, we can construct more complex and deep neural networks to solve different tasks.

This implementation includes the following contents:
+ [cnpy.cpp](cnpy.cpp) and [cnpy.h](cnpy.h), `git clone` from [Rogersce](https://github.com/rogersce/cnpy), are used to load `.npz` datasets.
+ [datasets.cpp](datasets.cpp) and [datasets.h](datasets.h) implemented that load `.npz` data (images and labels) into `torch::Tensor`. In our implementation, we encapsulated API of loading [MedMNIST](https://medmnist.com/) datasets and [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
+ [dataloader.cpp](dataloader.cpp) and [dataloader.h](dataloader.h) files, according the batch size and data category (`DATA_CAT::TRIN`, `DATA_CAT::VAL` and `DATA_CAT::TEST`), load images and labels into a `torch::Tensor` `std::vector`;
+ [train_model.cpp](train_model.cpp) and [train_model.h](train_model.h) files are used to define the convolution neural networks. At same time, it includes the training, valuation and test processes. This two files mainly used to train the model to get the parameters of weights and bias.
+ [he_utils.cpp](he_utils.cpp) and [he_utils.h](he_utils.h) files are used to encapsulate the leveled homomorphic encryption tools. And it also defines the `Cipher_Tensor` which looks like the `torch::Tensor`. But its element is the ciphertext polynomial of CKKS.
+ [he_layer.cpp](he_layer.cpp) and [he_layer.h](he_layer.h) files define the homomorphic layers, including homomorphic 2D convolution layer (`HEConv2dLayer`), homomorphic average layer (`HEAverage2dLayer`), homomorphic square activation layer (`HESquare`), and homomorphic fully connected layer (`HELinear`).
+ [conv_unit.cpp](conv_unit.cpp) and [conv_unit.h](conv_unit.h) test the efficiency of each layers in our homomorphic convolution unit.
+ [LoLa-Dense.cpp](LoLa_Dense.cpp) and [LoLa-Dense.h](LoLa_Dense.h) construct the homomorphic [LoLa-Dense](https://arxiv.org/pdf/1812.10659.pdf) network and test the efficiency of each homomorphic layers.
+ [CryptoNets.cpp](CryptoNets.cpp) and [CryptoNets.h](CryptoNets.h) construct the homomorphic [CryptoNets](https://dl.acm.org/doi/abs/10.5555/3045390.3045413) network and test the efficiency of each homomorphic layers.
+ [lenet.cpp](lenet.cpp) and [lenet.h](lenet.h) construct the homomorphic [LeNet](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=726791) network and test the efficiency of it each homomorphic layers.
+ [./experiment/conv_unit](experiment/conv_unit), [./experiment/loladense](experiment/loladense), [./experiment/cryptonets](experiment/cryptonets) and [./experiment/lenet](experiment/lenet) include the experimental results about conv_unit, LoLa-Dense, CryptoNets and LeNet separately.

### Environment Configuration
To use the APIs in our homomorphic convolution unit, you need install [*Libtorch*](https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.10.0%2Bcu113.zip) library and [*SEAL*](https://github.com/microsoft/SEAL) library (>3.7).
In our implementation, we use *CMake* tools to compile. If you want to compile by *CMake*, you only add the library dependency statement in [CMakeList.txt](CMakeLists.txt) to your *CMakeList.txt* file.