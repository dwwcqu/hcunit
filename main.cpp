#include<torch/torch.h>
#include "train_model.h"
#include "lenet.h"


int main() {
    using namespace dww;
    // train_mnist_model("mnist",10,256,0.001);
    for(const auto& fl : dww::channel_num){
        if(fl.first != "chestmnist" && fl.first != "retinamnist" && fl.second == 1){
             train_LeNet_model(fl.first, 5, 64, 0.001);
             LeNet_he_inference_test(fl.first);
        }
    }
    return 0;
}
