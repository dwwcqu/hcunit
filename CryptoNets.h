//
// Created by Dengweiwei on 2021/12/6.
//

#ifndef EXPERIMENT_CRYPTONETS_H
#define EXPERIMENT_CRYPTONETS_H
#include "he_layer.h"
namespace dww{
    struct CryptoNets : HEBaseModule{
        CryptoNets(const torch::Tensor& conv1_w,const torch::Tensor& conv1_b,
                      int64_t conv1_in,int64_t conv1_out,int64_t conv1_k,int64_t conv1_p,int64_t conv1_s,
                      int64_t pool1_k,int64_t pool1_p,int64_t pool1_s,
                      const torch::Tensor& conv2_w,const torch::Tensor& conv2_b,
                      int64_t conv2_in,int64_t conv2_out,int64_t conv2_k,int64_t conv2_p,int64_t conv2_s,
                      int64_t pool2_k,int64_t pool2_p,int64_t pool2_s,
                      const torch::Tensor& l1_w, const torch::Tensor& l1_b,
                      int64_t l1_in,int64_t l1_out,
                      const torch::Tensor& l2_w,const torch::Tensor& l2_b,
                      int64_t l2_in,int64_t l2_out);
        void forward(const torch::Tensor& input,HEWrapper& tools,Cipher_Tensor& output) override;
        HEConv2dLayer conv1,conv2;
        HEAverage2dLayer pool1,pool2;
        HELinear linear1,linear2;
        HESquare square;
        double conv_time = 0.0, pool_time = 0.0, square_time = 0.0, linear_time = 0.0;
        double enc_time = 0.0, dec_time = 0.0;
    };
    void CryptoNets_he_inference_test(const std::string& filename);
    std::ostream& operator<<(std::ostream& out,const CryptoNets& self);
}
#endif //EXPERIMENT_CRYPTONETS_H
