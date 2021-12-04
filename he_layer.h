//
// Created by Dengweiwei on 2021/11/19.
//

#ifndef EXPERIMENT_HE_LAYER_H
#define EXPERIMENT_HE_LAYER_H
#define SHAPE(N,K,S,P) (((N - K + 2*P) / S) + 1)
#include <vector>
#include <seal/seal.h>
#include <chrono>
#include "he_utils.h"
#include "dataloader.h"
using namespace seal;

namespace dww{
    using std::chrono::high_resolution_clock;
    /**
     * 深度学习每层运算的基类
     */
    class HEBaseLayer{
    public:
        /**
         * @param cipher    图像像素的密文形式
         * @param tools     同态加密各类操作封装对象
         * @param res       保存卷积后的结果
         * 根据该 2D 卷积层的输入通道数、输出通道数、卷积核、步长、填充等参数，返回输入密文对应的密文态卷积结果
         */
        virtual void forward(const dww::Cipher_Tensor& sour,dww::HEWrapper& tools,dww::Cipher_Tensor& dest) = 0;
    };


    /**
     * 深度学习中的 2D 同态卷积运算层
     */
    class HEConv2dLayer : public HEBaseLayer{
    public:
        HEConv2dLayer(std::vector<double> w,std::vector<double> b,int64_t in_c,int64_t out_c,int64_t win=3,int64_t pad=0,int64_t stri=1);
        HEConv2dLayer(const torch::Tensor& w,const torch::Tensor& b,int64_t in_c,int64_t out_c,int64_t win=3,int64_t pad=0,int64_t stri=1);
        void forward(const dww::Cipher_Tensor& sour,HEWrapper& tools,dww::Cipher_Tensor& dest) override;
        // 根据输入图片的 shape，确定输出的元素数量
        [[nodiscard]] int64_t out_numel(const std::vector<int64_t>& input_shape) const;
        // 根据输入图片的 shape，确定输出元素 shape
        [[nodiscard]] std::vector<int64_t> out_shape(const std::vector<int64_t>& input_shape) const;
        int64_t in_channel,out_channel;    // 输入、输出通道数
        int64_t windows,padding,stride;    // 卷积核、填充和步长
        std::vector<double> weights;        // 该卷积层的权重值
        std::vector<int64_t> w_shape;      // 该权重构成的 shape: N*M*K*K 的形式；N: 输入通道值;M: 输出通道值; K: 卷积核大小
        std::vector<double> bias;           // 偏值: shape 始终为 M*1
    private:
        /**
         * @param n         输入第几通道值
         * @param m         输出第几通道值
         * @param row       2D 卷积的 image 的行号
         * @param col       2D 卷积的 image 的列号
         * @param k         2D 卷积的核大小
         * @param H         image 的高度
         * @param W         image 的宽度
         * @return          返回对应卷积权重值的 vector, vector 按照从上大小、从左到右的顺序进行排列
         */
        [[nodiscard]] std::vector<double> get_one_available_window_at(int64_t n,int64_t m,int64_t row,int64_t col,int64_t k,int64_t H,int64_t W) const;
    };
    /**
     * 2D 同态平均池化
     */
    class HEAverage2dLayer : public HEBaseLayer{
    public:
        explicit HEAverage2dLayer(int64_t win,int64_t pad = 0,int64_t stri = 1) : windows(win),padding(pad),stride(stri)
        {
        }
        void forward(const dww::Cipher_Tensor& sour,HEWrapper& tools,dww::Cipher_Tensor& dest) override;
        [[nodiscard]] int64_t out_numel(const std::vector<int64_t>& input_shape) const;
        [[nodiscard]] std::vector<int64_t> out_shape(const std::vector<int64_t>& input_shape) const;
        int64_t windows,padding,stride;
    };

    class HESquare{
    public:
        HESquare() = default;
        void forward(dww::Cipher_Tensor& sour,HEWrapper& tools);
    };

    class HELinear : public HEBaseLayer{
    public:
        HELinear() = delete;
        HELinear(std::vector<double>& w,std::vector<double>& b,int64_t in,int64_t out);
        HELinear(const torch::Tensor& w,const torch::Tensor& b,int64_t in,int64_t out);
        void forward(const dww::Cipher_Tensor& sour,HEWrapper& tools,dww::Cipher_Tensor& dest) override;
        [[nodiscard]] int64_t out_numel() const;
        [[nodiscard]] std::vector<int64_t> out_shape() const;

        std::vector<double> weights;
        std::vector<double> bias;
        int64_t in_,out_;
    };
    /**
     *  同态卷积模型基类
     */
    struct HEBaseModule{
        virtual void forward(const torch::Tensor& input,HEWrapper& tools,Cipher_Tensor& output) = 0;
    };
    std::ostream& operator<<(std::ostream& out,const HELinear& self);
    std::ostream& operator<<(std::ostream& out,const HEConv2dLayer& self);
    std::ostream& operator<<(std::ostream& out,const HEAverage2dLayer& self);

}
#endif //EXPERIMENT_HE_LAYER_H
