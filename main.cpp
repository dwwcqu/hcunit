#include<torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include "train_model.h"
#include "conv_unit.h"
#include "CryptoNets.h"
#include "LoLa_Dense.h"
#include "lenet.h"


int main() {
    using namespace dww;
    int8_t cuda_index = 1;
    c10::cuda::CUDAGuard device_guard(cuda_index); //指定默认的显卡索引
    // train_mnist_model("mnist",10,256,0.001);
    for(const auto& fl : dww::channel_num){
        if(fl.first != "chestmnist" && fl.first != "retinamnist" && fl.second == 1){
//            train_1_channel_model(fl.first,3,64,0.001);
//            conv_unit_he_inference_test(fl.first,8192,25);
//            conv_unit_he_inference_test(fl.first,16384,30);

            train_CryptoNets_model(fl.first,3,64,0.001);
            CryptoNets_he_inference_test(fl.first);

            train_LoLaDense_model(fl.first,3,64,0.001);
            LoLaDense_he_inference_test(fl.first,8192,25);
            LoLaDense_he_inference_test(fl.first,16384,25);

            train_LeNet_model(fl.first, 3, 64, 0.001);
            LeNet_he_inference_test(fl.first);
        }
    }
    return 0;
}
