//
// Created by 邓维维 on 2021/12/6.
//

#ifndef EXPERIMENT_LOLA_DENSE_H
#define EXPERIMENT_LOLA_DENSE_H
#include "he_layer.h"
namespace dww{
    struct LoLaDense : HEBaseModule{
        LoLaDense(const torch::Tensor& conv_w,const torch::Tensor& conv_b,
                  int64_t conv_in,int64_t conv_out,int64_t conv_k,int64_t conv_p,int64_t conv_s,
                  const torch::Tensor& l1_w, const torch::Tensor& l1_b,
                  int64_t l1_in,int64_t l1_out,
                  const torch::Tensor& l2_w,const torch::Tensor& l2_b,
                  int64_t l2_in,int64_t l2_out);
        void forward(const torch::Tensor& input,HEWrapper& tools,Cipher_Tensor& output) override;
        HEConv2dLayer conv;
        HELinear linear1,linear2;
        HESquare square;
        double conv_time = 0.0,square_time = 0.0,linear_time = 0.0;
        double enc_time = 0.0,dec_time = 0.0;
    };
    void LoLaDense_he_inference_test(const std::string& filename,int64_t poly_d,int64_t scale);
    std::ostream& operator<<(std::ostream& out,const LoLaDense& self);
}

#endif //EXPERIMENT_LOLA_DENSE_H
