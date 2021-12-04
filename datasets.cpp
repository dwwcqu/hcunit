//
// Created by 邓维维 on 2021/11/13.
//

#include "datasets.h"
#include <iostream>

void dww::print_all_dataset_info() {
    int count = 1;
    for(const auto& [dataname,num] : channel_num){
        std::cout << "Dataset " << count << " : " << dataname << '\n';
        std::cout << "\tChannel Numbers ---> " << num << " <---\n";
        std::cout << "\tLabel Numbers ---> " << label_num.at(dataname) << " <---\n";
        std::cout << "\tDetail Labels Information : \n";
        for(const auto& [id,info] : labels_info.at(dataname)){
            std::cout << "\t\tLabel " << id << " ---> " << info << " <---\n";
        }
        std::cout << "\tDataset Samples Information : \n";
        for(const auto& [samp,num] : samples.at(dataname)){
            std::cout << "\t\tCategory " << samp << " ---> " << num << " <---\n";
        }
        ++count;
    }
}

void dww::print_all_dataset_samples_info(){
    using namespace cnpy;
    for(const auto& pair : channel_num){
        std::string path = "../medmnist/";
        path += pair.first;
        path += ".npz";
        npz_t datas = npz_load(path);
        std::cout << "Datasets " << pair.first << "'s Informations: \n";
        for(auto& [name,array] : datas){
            std::cout << "\tSub-Datasets Name " << name << "\tShape : " << array.shape << '\n';
        }
    }
}

torch::Tensor dww::get_images_tensor_from_NpyArray(cnpy::NpyArray& npy){
    vector<int64_t> shape(npy.shape.begin(),npy.shape.end());
    torch::Tensor result = torch::tensor(npy.as_vec<uint8_t>()).to(torch::kFloat64);
    result = result.reshape(shape);
    if(shape.size() == 4)
        return result.permute({0,3,1,2});
    else
        return result.unsqueeze(1);
}
torch::Tensor dww::get_labels_tensor_from_NpyArray(cnpy::NpyArray& npy){
    vector<int64_t> shape(npy.shape.begin(),npy.shape.end());
    torch::Tensor result = torch::tensor(npy.as_vec<uint8_t>());
    if(shape[1] > 1)
        return result.reshape(shape).to(torch::kFloat64);
    return result.to(torch::kInt64);
}

dww::MedDataSet::MedDataSet(const std::string& filename){
    std::string path = ADD_MED_SUFFIX(filename);
    cnpy::npz_t alldata = cnpy::npz_load(path);

    train_images = get_images_tensor_from_NpyArray(alldata["train_images"]);
    train_labels = get_labels_tensor_from_NpyArray(alldata["train_labels"]);

    val_images = get_images_tensor_from_NpyArray(alldata["val_images"]);
    val_labels = get_labels_tensor_from_NpyArray(alldata["val_labels"]);

    test_images = get_images_tensor_from_NpyArray(alldata["test_images"]);
    test_labels = get_labels_tensor_from_NpyArray(alldata["test_labels"]);
}

dww::MNISTDataSet::MNISTDataSet(const std::string& filename){
    std::string path = ADD_MNIST_SUFFIX(filename);
    cnpy::npz_t alldata = cnpy::npz_load(path);

    train_images = get_images_tensor_from_NpyArray(alldata["x_train"]);
    train_labels = get_labels_tensor_from_NpyArray(alldata["y_train"]);

    test_images = get_images_tensor_from_NpyArray(alldata["x_test"]);
    test_labels = get_labels_tensor_from_NpyArray(alldata["y_test"]);
}