//
// Created by Dengweiwei on 2021/11/16.
//
#include "train_model.h"
using namespace std::chrono;
dww::TASK_CAT dww::get_task_cat(const std::string &filename) {
    const std::string& temp = task.at(filename);
    if(temp == "multi-class")
        return MULTI_CLASS;
    else if(temp == "multi-label, binary-class")
        return MULTI_LABEL_BINARY_CLASS;
    else if(temp == "binary-class")
        return BINARY_CLASS;
    else
        return ORDINAL_REGRESSION;
}


void dww::train_1_channel_model(const std::string& filename,int64_t epoch,int64_t batch,double lr){
    std::string path = "../parameters/" + filename;
    std::ofstream log_file(path,ios_base::trunc);
    assert(log_file.is_open() && "File doesn't open!");
    log_file << "Dataset " << filename << " : " << '\n';
    log_file << "Train Samples : " << samples.at(filename).at("train")
            << " | Val Samples : " << samples.at(filename).at("val")
            << " | Test Samples : " << samples.at(filename).at("test") << '\n';
    using std::chrono::high_resolution_clock;
    TORCH_MODULE(Test_1_1_CHAN_Model);
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    // 根据数据集 filename 的标签个数，定义针对该数据集的卷积神经网络
    int64_t label_sz = label_num.at(filename);
    bool is_multi_label_binary_class = task.at(filename) == "multi-label, binary-class";
    Test_1_1_CHAN_Model OneCNet(label_sz);
    log_file << "Network Model: ";
    log_file << OneCNet << '\n';
    // 使用 GPU 进行训练
    OneCNet->to(device);
    // 获取 filename 数据集中的 TRIN、VAL、TEST 数据集
    dww::MedDataSet all_data(filename);
    // 获取训练数据集
    dww::MedDataSetLoader train_data(all_data,dww::DATA_CAT::TRIN,batch);
    // 获取 Valuation 数据集
    dww::MedDataSetLoader val_data(all_data,dww::DATA_CAT::VAL,2 * batch);
    // 测试数据集用来进行 Inference 操作

    // 使用随机梯度下降优化算法
    torch::optim::SGD optimizer(OneCNet->parameters(),torch::optim::SGDOptions(lr));
    high_resolution_clock::time_point start = high_resolution_clock::now();
    double total_acc = 0.0;
    for(size_t ep = 1; ep <= epoch; ++ep) {
        log_file << "\t Epoch : " << ep;
        train_1_channel(*OneCNet,optimizer,train_data,device,is_multi_label_binary_class);
        double acc = val_1_channel(*OneCNet,val_data,device,is_multi_label_binary_class);
        total_acc += acc;
        log_file << "\tAcc : " << acc*100 << "%\n";
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    double time_consume = duration_cast<duration<double>>(end - start).count();
    total_acc /= epoch;
    log_file << "\tEpoch: " << epoch
            <<"\tBatch: " << batch
            << "\tlr: " << lr
            << "\tTime Consume: " << time_consume << " seconds"
            << "\tAverage Acc: " << total_acc * 100 << "%\n";
    std::string param_path = "../model/";
    for(auto pair : OneCNet->named_parameters()){
        std::string sub_path = filename;
        sub_path += "/conv_unit/";
        std::string temp = pair.key();
        size_t pos = temp.find_first_of('.');
        temp[pos] = '_';
        sub_path += temp;
        sub_path += ".pt";
        torch::save(pair.value(),param_path + sub_path);
    }
    log_file << "\n\n";
    log_file.flush();
    log_file.close();
}

void dww::train_mnist_model(const std::string& filename,int64_t epoch,int64_t batch,double lr){
    TORCH_MODULE(MNISTSimpleModel);
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // 根据数据集 filename 的标签个数，定义针对该数据集的卷积神经网络
    int64_t label_sz = 10;
    MNISTSimpleModel OneCNet(label_sz);
    // 使用 GPU 进行训练
    OneCNet->to(device);
    // 获取 MNIST 数据中的 TRIN、TEST 数据集
    dww::MNISTDataSet all_data("mnist");
    // 获取 TRAIN 数据集
    dww::MNISTDataSetLoader train_data(all_data,dww::DATA_CAT::TRIN,batch);
    // 获取 TEST 数据集
    dww::MNISTDataSetLoader test_data(all_data,dww::DATA_CAT::TEST,2 * batch);
    // 使用随机梯度下降优化算法
    torch::optim::SGD optimizer(OneCNet->parameters(),torch::optim::SGDOptions(lr));
    for(int64_t ep = 1; ep <= epoch; ++ep){
        std::cout << "Epoch : " << ep << "\n";
        train_1_channel(*OneCNet,optimizer,train_data,device);
        double acc = val_1_channel(*OneCNet,test_data,device);
        std::cout << "\tAcc : " << acc*100 << "%\n";
    }
}


void dww::train_1_channel(BaseModelImpl& model,
                            torch::optim::Optimizer& optimizer,
                            dww::BaseDataLoader& train_data,
                            const torch::Device& device,bool is_multi_label_binary_class){
    int64_t batch_num = train_data.get_batch_num();
    model.train();
    for(int64_t sz = 0; sz < batch_num; ++sz){
        optimizer.zero_grad();
        torch::Tensor image = train_data.images[sz].to(device);
        torch::Tensor label = train_data.labels[sz].to(device);
        torch::Tensor predict = model.forward(image);
        torch::Tensor loss = is_multi_label_binary_class ? torch::binary_cross_entropy_with_logits(predict,label) : torch::cross_entropy_loss(predict,label);
        loss.backward();
        optimizer.step();
        std::cout << "\t[" << sz+1 << "/" << batch_num << "],Loss : " << loss.item<double>() << "\n";
    }
}
double dww::val_1_channel(BaseModelImpl& model,dww::BaseDataLoader& val_data,
                          const torch::Device& device,bool is_multi_label_binary_class){
    int64_t samples_num = val_data.samples_num;
    int64_t batch_num = val_data.get_batch_num();
    int64_t correct = 0;
    model.eval();
    for(int64_t sz = 0; sz < batch_num; ++sz){
        torch::Tensor image = val_data.images[sz].to(device);
        torch::Tensor label = val_data.labels[sz].to(device);
        int64_t batch_size = label.size(0);
        torch::Tensor predict = model.forward(image);
        torch::Tensor cmp;
        cmp = is_multi_label_binary_class ? predict.argmax(1) - label.argmax(1)  : predict.argmax(1) - label;
        correct += (batch_size - cmp.count_nonzero().item<int64_t>());
    }
    return static_cast<double>(correct) / static_cast<double>(samples_num);
}


void dww::train_LeNet_model(const std::string& filename,int64_t epoch,int64_t batch,double lr){
    std::string path = "../parameters/lenet/" + filename;
    std::ofstream log_file(path,ios_base::trunc);
    assert(log_file.is_open() && "File doesn't open!");
    log_file << "Dataset " << filename << " : " << '\n';
    log_file << "Train Samples : " << samples.at(filename).at("train")
             << " | Val Samples : " << samples.at(filename).at("val")
             << " | Test Samples : " << samples.at(filename).at("test") << '\n';
    using std::chrono::high_resolution_clock;
    TORCH_MODULE(LeNet);
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    // 根据数据集 filename 的标签个数，定义针对该数据集的卷积神经网络
    int64_t label_sz = label_num.at(filename);
    bool is_multi_label_binary_class = task.at(filename) == "multi-label, binary-class";
    LeNet OneCNet(label_sz);
    log_file << "Network Model: ";
    log_file << OneCNet << '\n';
    // 使用 GPU 进行训练
    OneCNet->to(device);
    // 获取 filename 数据集中的 TRIN、VAL、TEST 数据集
    dww::MedDataSet all_data(filename);
    // 获取训练数据集
    dww::MedDataSetLoader train_data(all_data,dww::DATA_CAT::TRIN,batch);
    // 获取 Valuation 数据集
    dww::MedDataSetLoader val_data(all_data,dww::DATA_CAT::VAL,2 * batch);
    // 测试数据集用来进行 Inference 操作

    // 使用随机梯度下降优化算法
    torch::optim::SGD optimizer(OneCNet->parameters(),torch::optim::SGDOptions(lr));
    high_resolution_clock::time_point start = high_resolution_clock::now();
    double total_acc = 0.0;
    for(size_t ep = 1; ep <= epoch; ++ep) {
        log_file << "\t Epoch : " << ep;
        train_1_channel(*OneCNet,optimizer,train_data,device,is_multi_label_binary_class);
        double acc = val_1_channel(*OneCNet,val_data,device,is_multi_label_binary_class);
        total_acc += acc;
        log_file << "\tAcc : " << acc*100 << "%\n";
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    double time_consume = duration_cast<duration<double>>(end - start).count();
    total_acc /= epoch;
    log_file << "\tEpoch: " << epoch
             <<"\tBatch: " << batch
             << "\tlr: " << lr
             << "\tTime Consume: " << time_consume << " seconds"
             << "\tAverage Acc: " << total_acc * 100 << "%\n";
    std::string param_path = "../model/lenet/";
    for(auto pair : OneCNet->named_parameters()){
        std::string sub_path = filename;
        sub_path += "/";
        std::string temp = pair.key();
        size_t pos = temp.find_first_of('.');
        temp[pos] = '_';
        sub_path += temp;
        sub_path += ".pt";
        torch::save(pair.value(),param_path + sub_path);
    }
    log_file << "\n\n";
    log_file.flush();
    log_file.close();
}