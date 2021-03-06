//
// Created by dengweiwei on 2021/11/12.
//

#ifndef EXPERIMENT_TRAIN_MODEL_H
#define EXPERIMENT_TRAIN_MODEL_H
#include <chrono>
#include <torch/torch.h>
#include <fstream>
#include "dataloader.h"
#include <string>
//dww namespace begin
namespace dww{
    enum TASK_CAT{
        MULTI_LABEL_BINARY_CLASS = 0,
        MULTI_CLASS,
        BINARY_CLASS,
        ORDINAL_REGRESSION
    };

    TASK_CAT get_task_cat(const std::string& filename);
    struct BaseModelImpl : public torch::nn::Module{
        virtual torch::Tensor forward(const torch::Tensor& input) = 0;
    };
    TORCH_MODULE(BaseModel);

    struct Med1CHANNELNetImpl : BaseModelImpl{
        Med1CHANNELNetImpl(int64_t num_labels)
        :
        conv1(torch::nn::Conv2dOptions(1,5,{5,5}).stride(2).padding(1)),
        pool1(torch::nn::AvgPool2dOptions({3,3}).stride(1).padding(1)),
        conv2(torch::nn::Conv2dOptions(5,50,{5,5}).stride(2)),
        pool2(torch::nn::AvgPool2dOptions({3,3}).stride(1).padding(1)),
        linear1(torch::nn::LinearOptions(50*5*5,1*100)),
        linear2(torch::nn::LinearOptions(1*100,num_labels))
        {
            register_module("conv1",conv1);
            register_module("conv2",conv2);
            register_module("pool1",pool1);
            register_module("pool2",pool2);
            register_module("linear1",linear1);
            register_module("linear2",linear2);
            register_module("relu1",relu1);
            register_module("relu2",relu2);
        }
        torch::Tensor forward(const torch::Tensor& input) override{
            torch::Tensor result = conv1(input);
            result = torch::relu(result);
            result = pool1(result);
            result = conv2(result);
            result = pool2(result);
            result = torch::nn::Flatten()(result);
            result = linear1(result);
            result = torch::relu(result);
            result = linear2(result);
            return result;
        }
        torch::nn::Conv2d conv1,conv2;
        torch::nn::AvgPool2d pool1,pool2;
        torch::nn::ReLU relu1,relu2;
        torch::nn::Linear linear1,linear2;
    };
    struct Med3CHANNELNetImpl : BaseModelImpl{
        Med3CHANNELNetImpl(int64_t num_labels)
                :
                conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(3,128,{3,3}).padding(1))),
                pool1(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({2,2}).stride(2))),
                conv2(torch::nn::Conv2d(torch::nn::Conv2dOptions(128,83,{3,3}))),
                pool2(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({2,2}).stride(2))),
                conv3(torch::nn::Conv2d(torch::nn::Conv2dOptions(83,163,{3,3}))),
                pool3(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({2,2}).stride(1))),
                linear1(torch::nn::Linear(torch::nn::LinearOptions(163*3*3,1024))),
                linear2(torch::nn::Linear(torch::nn::LinearOptions(1024,num_labels))),
                sigmoid(torch::nn::Sigmoid())
        {
            register_module("conv1",conv1);
            register_module("conv2",conv2);
            register_module("conv3",conv3);
            register_module("pool1",pool1);
            register_module("pool2",pool2);
            register_module("pool3",pool3);
            register_module("linear1",linear1);
            register_module("linear2",linear2);
            register_module("sigmoid",sigmoid);
        }
        torch::Tensor forward(const torch::Tensor& input) override{
            torch::Tensor result = conv1(input);
            //std::cout << "conv1(input) = " << input.strides() << '\n';
            result = pool1(result);
            //std::cout << "pool1(input) = " <<  input.strides() << std::endl;
            result = conv2(result);
            //std::cout << "conv2(input) = " <<  input.strides() << std::endl;
            result.square_();
            //std::cout << "square(input) = " << input.strides() << '\n';
            result = pool2(result);
            //std::cout << "pool2(input) = " << input.strides() << '\n';
            result = conv3(result);
            //std::cout << "conv3(input) = " <<  input.strides() << std::endl;
            result = pool3(result.square());
            //std::cout << "pool3(input) = " <<  input.strides() << std::endl;
            result = torch::nn::Flatten()(result);
            //std::cout << "flatten(input) = " << input.strides() << '\n';
            result = linear2(linear1(result));
            result = sigmoid(result);
            return result;
        }
        torch::nn::Conv2d conv1,conv2,conv3;
        torch::nn::AvgPool2d pool1,pool2,pool3;
        torch::nn::Linear linear1,linear2;
        torch::nn::Sigmoid sigmoid;
    };
    struct MNISTModelImpl : BaseModelImpl {
        MNISTModelImpl():
            conv1(torch::nn::Conv2d(torch::nn::Conv2dOptions(1,6,5))),
            conv2(torch::nn::Conv2d(torch::nn::Conv2dOptions(6,16,5))),
            max1(torch::nn::MaxPool2dOptions(2).stride(2)),
            max2(torch::nn::MaxPool2dOptions(2).stride(2)),
            lin1(torch::nn::Linear(torch::nn::LinearOptions(16*4*4,120))),
            lin2(torch::nn::Linear(torch::nn::LinearOptions(120,84))),
            lin3(torch::nn::Linear(torch::nn::LinearOptions(84,10)))
        {
            register_module("conv1",conv1);
            register_module("conv2",conv2);
            register_module("maxpool1",max1);
            register_module("maxpool2",max2);
            register_module("linear1",lin1);
            register_module("linear2",lin2);
            register_module("linear3",lin3);
        }
        torch::Tensor forward(const torch::Tensor& input) override{
            torch::Tensor result = torch::sigmoid(conv1(input));
            result = conv2(max1(result));
            result = torch::sigmoid(result);
            result = max2(result);
            result = torch::flatten(result,1);
            result = torch::sigmoid(lin1(result));
            result = torch::sigmoid(lin2(result));
            return lin3(result);
        }
        torch::nn::Conv2d conv1,conv2;
        torch::nn::MaxPool2d max1,max2;
        torch::nn::Linear lin1,lin2,lin3;
    };
    struct MNISTSimpleModelImpl : BaseModelImpl{
        MNISTSimpleModelImpl(int64_t label_num):
        linear1(torch::nn::LinearOptions(28*28,512)),
        linear2(torch::nn::LinearOptions(512,512)),
        linear3(torch::nn::LinearOptions(512,label_num))
        {
            register_module("linear1",linear1);
            register_module("linear2,",linear2);
            register_module("linear3",linear3);
        }
        torch::Tensor forward(const torch::Tensor& input) override{
            torch::Tensor result = torch::flatten(input,1);
            result = torch::leaky_relu(linear1(result));
            result = torch::leaky_relu(linear2(result));
            return linear3(result);
        }
        torch::nn::Linear linear1,linear2,linear3;
    };


    struct Test_1_1_CHAN_ModelImpl : BaseModelImpl{
        explicit Test_1_1_CHAN_ModelImpl(int64_t num_label)
        :
        conv(torch::nn::Conv2dOptions(1,8,3).padding(1).stride(1)),
        pool(torch::nn::AvgPool2dOptions(2).stride(2).padding(1)),
        linear1(torch::nn::LinearOptions(8*15*15,64)),
        linear2(torch::nn::LinearOptions(64,num_label))
        {
            register_module("conv",conv);
            conv->to(torch::kFloat64);
            register_module("leaky-LeRU",lrelu);
            lrelu->to(torch::kFloat64);
            register_module("pool",pool);
            pool->to(torch::kFloat64);
            register_module("linear1",linear1);
            linear1->to(torch::kFloat64);
            register_module("linear2",linear2);
            linear2->to(torch::kFloat64);
        }
        torch::Tensor forward(const torch::Tensor& input) override{
            torch::Tensor result = lrelu(pool(conv(input)));
            result = torch::flatten(result,1);
            result = linear2(linear1(result));
            return result;
        }
        torch::nn::Conv2d conv;
        torch::nn::AvgPool2d pool;
        torch::nn::LeakyReLU lrelu;
        torch::nn::Linear linear1;
        torch::nn::Linear linear2;
    };



    /**
     * @param model                         ??????
     * @param optimizer                     ?????????
     * @param train_data                    ????????????????????????
     * @param device                        kCPU/kCUDA
     * @param is_multi_label_binary_class   ???????????????????????????????????????
     */
    void train_1_channel(BaseModelImpl& model,
                           torch::optim::Optimizer& optimizer,
                           dww::BaseDataLoader& train_data,
                           const torch::Device& device,bool is_multi_label_binary_class = false);
    /**
     * @param model                         ??????
     * @param val_data                      ???????????????
     * @param device                        kCPU/kCUDA
     * @param is_multi_label_binary_class   ???????????????????????????????????????
     * @return                              ???????????????
     */
    double val_1_channel(BaseModelImpl& model,dww::BaseDataLoader& val_data,const torch::Device& device,bool is_multi_label_binary_class = false);
    /**
     * @param filename  ????????????????????? filename ?????????????????????????????????filename ??????????????????????????? 1
     * @param epoch     epoch ???????????????????????????????????????
     * @param batch     batch ????????????????????????????????????
     * @param lr        ?????????
     */
    void train_1_channel_model(const std::string& filename,int64_t epoch,int64_t batch,double lr = 0.01);


    void train_3_channel(BaseModelImpl& model,
                           torch::optim::Optimizer& optimizer,
                         dww::BaseDataLoader& train_data,
                           const torch::Device& device);
    double val_3_channel(BaseModelImpl& model,dww::BaseDataLoader& val_data,const torch::Device& device);

    // ????????????????????? 3 ?????????
    // ?????????????????????????????? Med3CHANNELNETImpl
    void train_3_channel_model(const std::string& filename,int64_t epoch,int64_t batch,double lr);


    void train_mnist_model(const std::string& filename,int64_t epoch,int64_t batch,double lr = 0.01);



    struct LeNetImpl : BaseModelImpl{
        LeNetImpl(int64_t label_num) :
        conv1(torch::nn::Conv2dOptions(1,6,{3,3})),
        pool1(torch::nn::AvgPool2dOptions({2,2}).stride(2)),
        conv2(torch::nn::Conv2dOptions(6,16,{3,3})),
        pool2(torch::nn::AvgPool2dOptions({2,2}).stride(2)),
        conv3(torch::nn::Conv2dOptions(16,120,{3,3})),
        linear1(torch::nn::LinearOptions(120*3*3,84)),
        linear2(torch::nn::LinearOptions(84,label_num)),
        model_name("lenet")
        {
            register_module("conv1",conv1);
            conv1->to(torch::kFloat64);
            register_module("pool1",pool1);
            register_module("conv2",conv2);
            conv2->to(torch::kFloat64);
            register_module("pool2",pool2);
            register_module("conv3",conv3);
            conv3->to(torch::kFloat64);
            register_module("linear1",linear1);
            linear1->to(torch::kFloat64);
            register_module("linear2",linear2);
            linear2->to(torch::kFloat64);
        }
        torch::Tensor forward(const torch::Tensor& input) override{
            torch::Tensor output = torch::relu(pool1(conv1(input)));
            output = torch::relu(pool2(conv2(output)));
            output = conv3(output);
            output = output.flatten(1);
            output = linear2(linear1(output));
            return output;
        }
        torch::nn::Conv2d conv1,conv2,conv3;
        torch::nn::AvgPool2d pool1,pool2;
        torch::nn::LeakyReLU relu;
        torch::nn::Linear linear1,linear2;
        std::string model_name;
    };
    /**
     * @param filename
     * @param epoch
     * @param batch
     * @param lr
     * ?????? LeNet ??????
     */
    [[maybe_unused]] void train_LeNet_model(const std::string& filename,int64_t epoch,int64_t batch,double lr = 0.01);



    struct CryptoNetsImpl : BaseModelImpl{
        CryptoNetsImpl(int64_t label_num):
        conv1(torch::nn::Conv2dOptions(1,5,{5,5}).stride(2).padding(1)),
        conv2(torch::nn::Conv2dOptions(5,50,{5,5}).stride(2)),
        pool1(torch::nn::AvgPool2dOptions({3,3}).padding(1).stride(1)),
        pool2(torch::nn::AvgPool2dOptions({3,3}).padding(1).stride(1)),
        linear1(torch::nn::LinearOptions(50*5*5,100)),
        linear2(torch::nn::LinearOptions(100,label_num))
        {
            register_module("conv1",conv1);
            conv1->to(torch::kFloat64);
            register_module("pool1",pool1);
            register_module("conv2",conv2);
            conv2->to(torch::kFloat64);
            register_module("pool2",pool2);
            register_module("linear1",linear1);
            linear1->to(torch::kFloat64);
            register_module("linear2",linear2);
            linear2->to(torch::kFloat64);
        }
        torch::Tensor forward(const torch::Tensor& input) override{
            torch::Tensor output = torch::leaky_relu(conv1(input));
            output = pool1(output);
            output = conv2(output);
            output = pool2(output).flatten(1);
            output = torch::leaky_relu(linear1(output));
            output = linear2(output);
            return output;
        }
        torch::nn::Conv2d conv1,conv2;
        torch::nn::AvgPool2d pool1,pool2;
        torch::nn::Linear linear1,linear2;
    };
    [[maybe_unused]] void train_CryptoNets_model(const std::string& filename,int64_t epoch,int64_t batch,double lr = 0.01);

    struct LoLaDenseImpl : BaseModelImpl{
        LoLaDenseImpl(int64_t label_num)
        :
        conv(torch::nn::Conv2dOptions(1,5,{5,5}).stride(2).padding(2)),
        linear1(torch::nn::LinearOptions(5*14*14,100)),
        linear2(torch::nn::LinearOptions(100,label_num))
        {
            register_module("conv",conv);
            conv->to(torch::kFloat64);
            register_module("linear1",linear1);
            linear1->to(torch::kFloat64);
            register_module("linear2",linear2);
            linear2->to(torch::kFloat64);
        }
        torch::Tensor forward(const torch::Tensor& input) override{
            torch::Tensor output = torch::leaky_relu(conv(input));
            output = output.flatten(1);
            output = torch::leaky_relu(linear1(output));
            output = linear2(output);
            return output;
        }
        torch::nn::Conv2d conv;
        torch::nn::Linear linear1,linear2;
    };
    [[maybe_unused]] void train_LoLaDense_model(const std::string& filename,int64_t epoch,int64_t batch,double lr = 0.01);
}   // namespace end

#endif //EXPERIMENT_TRAIN_MODEL_H
