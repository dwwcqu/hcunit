//
// Created by 14128 on 2021/12/3.
//

#ifndef EXPERIMENT_CONV_UNIT_H
#define EXPERIMENT_CONV_UNIT_H
#include "he_layer.h"
namespace dww {
    struct HETest_1_1_CHAN_Model : HEBaseModule{
        HETest_1_1_CHAN_Model(std::vector<double>& conv2d_w,std::vector<double>& conv2d_b,
                              int64_t conv2d_in,int64_t conv2d_out,int64_t conv2d_k,int64_t conv2d_p,int64_t conv2d_s,
                              int64_t pool2d_k,int64_t pool2d_p,int64_t pool2d_s,
                              std::vector<double>& l1_w, std::vector<double>& l1_b,
                              int64_t l1_in,int64_t l1_out,
                              std::vector<double>& l2_w,std::vector<double>& l2_b,
                              int64_t l2_in,int64_t l2_out);

        void forward(const torch::Tensor& input,HEWrapper& tools,Cipher_Tensor& output) override;
        HEConv2dLayer conv2d;
        HEAverage2dLayer pool2d;
        HESquare square;;
        HELinear linear1,linear2;
        double time_conv = 0.0,time_relu = 0.0,time_pool = 0.0,time_l1 = 0.0,time_l2 = 0.0;
        double time_enc = 0.0,time_dec = 0.0;
    };

    void conv_unit_he_inference_test(const std::string& filename);
}
#endif //EXPERIMENT_CONV_UNIT_H
