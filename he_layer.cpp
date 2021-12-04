//
// Created by Dengweiwei on 2021/11/19.
//
#include "he_layer.h"
#include <cassert>
dww::HEConv2dLayer::HEConv2dLayer(std::vector<double> w,std::vector<double> b,
                                  int64_t in_c,int64_t out_c,
                                  int64_t win,int64_t pad,
                                  int64_t stri) :
    weights(move(w)),bias(move(b)),in_channel(in_c),out_channel(out_c),windows(win),padding(pad),stride(stri),w_shape({in_c,out_c,win,win}){

    // 卷积窗口的数量为 in_channel * out_channel, 其对应的权重参数为 win * win * out_c * in_c
    // assert(win * win * out_c * in_c == w.size() && b.size() == out_c && "The Number of Weights or Bias is not Correct!");
    // 窗口的要比 padding 大，否则无意义
    assert(windows > padding && "The values of windows and padding is not correct, windows need bigger than padding!");
    assert(w_shape.size() == 4 && "The shape of Windows is N * M * W * W!");
}

void dww::HEConv2dLayer::forward(const dww::Cipher_Tensor& sour,HEWrapper& tools,dww::Cipher_Tensor& dest){
    // 2D 卷积的输入形式为 N * H * W
    assert(sour.shape.size() == 3 && "The input image size is not correct. The input form is n * h * w!");
    // 输入的通道值必须和 in_channel 值一样，否则无法进行卷积
    assert(sour.shape[0] == in_channel && "The input channel number is not equal to in_channel value set!");
    int64_t N = in_channel,M = out_channel,H = sour.shape[1],W = sour.shape[2],K = windows,P = padding,S = stride;
    int64_t n_H = SHAPE(H,K,S,P);
    int64_t n_W = SHAPE(W,K,S,P);
    int64_t id = 0;
    for(int64_t m = 0; m < M; ++m){
        for(int64_t r = -P; r < H + P - K + 1; r += S ){
            for(int64_t c = -P; c < W + P - K + 1; c += S) {
                std::vector<double> weight(move(get_one_available_window_at(0,m,r,c,K,H,W)));
                sour.conv2d_one_window(0,r,c,K,weight,dest.value[id],tools);
                for (int64_t n = 1; n < N; ++n) {
                    // 获取权重值
                    std::vector<double> wt(get_one_available_window_at(n,m,r,c,K,H,W));
                    seal::Ciphertext temp;
                    sour.conv2d_one_window(n,r,c,K,wt,temp,tools);
                    tools.evl.add_inplace(dest.value[id],temp);
                }
                // 获取 bias 的明文多项式
                seal::Plaintext plain_b;
                tools.get_plaintext_of(bias[m],plain_b);
                // 为了确保两者能相加，必须使其有相同的参数链
                tools.evl.mod_switch_to_inplace(plain_b,dest.value[id].parms_id());
                // 同时，两者也必须有相同的 scale 值
                dest.value[id].scale() = plain_b.scale();
                // 相加
                tools.evl.add_plain_inplace(dest.value[id],plain_b);
                ++id;
            }
        }
    }
    assert(id == M*n_H*n_W && "Convolution process is not correct! The number of result is not equal to out_channel * ((H-K+2p)/S+1) * ((W-K+2p)/S+1)");
}
int64_t dww::HEConv2dLayer::out_numel(const std::vector<int64_t>& input_shape) const{
    assert(input_shape.size() == 3 && "The input's shape is not correct, it must be N * H * W!");
    int64_t n_H = SHAPE(input_shape[1],windows,stride,padding);
    int64_t n_W = SHAPE(input_shape[2],windows,stride,padding);
    return out_channel * n_H * n_W;
}
std::vector<int64_t> dww::HEConv2dLayer::out_shape(const std::vector<int64_t>& input_shape) const{
    assert(input_shape.size() == 3 && "The input's shape is not correct, it must be N * H * W!");
    int64_t n_H = SHAPE(input_shape[1],windows,stride,padding);
    int64_t n_W = SHAPE(input_shape[2],windows,stride,padding);
    return {out_channel,n_H,n_W};
}



std::vector<double>
dww::HEConv2dLayer::get_one_available_window_at(int64_t n, int64_t m, int64_t row, int64_t col, int64_t k,int64_t H,int64_t W) const {
    assert(n < in_channel && "The input channel value must smaller than in_channel!");
    assert(m < out_channel && "The output channel value must smaller than out_channel");
    size_t start_channel = n * out_channel * windows * windows;
    int64_t x_1 = std::max(0L,row);
    int64_t y_1 = std::max(0L,col);
    int64_t x_2 = std::min(H - 1,row + k - 1);
    int64_t y_2 = std::min(W - 1,col + k - 1);
    std::pair<int,int> left_up(x_1 - row,y_1 - col);
    std::pair<int,int> right_down(x_2 - row,y_2 - col);
    std::vector<double> res;
    for(size_t i = left_up.first; i <= right_down.first; ++i){
        for(size_t j = left_up.second; j <= right_down.second; ++j){
            size_t index = start_channel + m*k*k + k*i + j;
            res.push_back(weights[index]);
        }
    }
    return res;
}

dww::HEConv2dLayer::HEConv2dLayer(const torch::Tensor& w, const torch::Tensor& b, int64_t in_c, int64_t out_c, int64_t win,
                                  int64_t pad, int64_t stri)
                                  :
                                  weights(w.data_ptr<double>(),w.data_ptr<double>() + w.numel()),
                                  bias(b.data_ptr<double>(),b.data_ptr<double>() + b.numel()),
                                  in_channel(in_c),out_channel(out_c),windows(win),padding(pad),stride(stri),w_shape({in_c,out_c,win,win})
                                  {

}

std::ostream& dww::operator<<(std::ostream& out,const HEConv2dLayer& self) {
    out << "\tIn channel = " << self.in_channel << "; Out Channel = " << self.out_channel << "; Windows = " << self.windows << '\n'
        << "\t\t; Padding = " << self.padding << "; Stride = " << self.stride << '\n';
    return out;
}

void dww::HEAverage2dLayer::forward(const dww::Cipher_Tensor& sour,HEWrapper& tools,dww::Cipher_Tensor& dest) {
    // 2D 平均卷积的输入形式为 N * H * W
    assert(sour.shape.size() == 3 && "The input image size is not correct. The input form is n * h * w!");
    int64_t M = sour.shape[0],H = sour.shape[1], W = sour.shape[2], K = windows, P = padding,S = stride;
    int64_t n_H = SHAPE(H,K,S,P);
    int64_t n_W = SHAPE(W,K,S,P);
    int64_t id = 0;
    for(int64_t m = 0; m < M; ++m) {
        for (int64_t r = -P; r < H + P - K + 1; r += S) {
            for (int64_t c = -P; c < W + P - K + 1; c += S) {
                sour.average2d_one_window(m,r,c,K,dest.value[id++],tools);
            }
        }
    }
    assert(id == M*n_H*n_W && "Convolution process is not correct! The number of result is not equal to out_channel * ((H-K+2p)/S+1) * ((W-K+2p)/S+1)");
}
int64_t dww::HEAverage2dLayer::out_numel(const std::vector<int64_t>& input_shape) const{
    assert(input_shape.size() == 3 && "The input's shape is not correct, it must be N * H * W!");
    int64_t n_H = SHAPE(input_shape[1],windows,stride,padding);
    int64_t n_W = SHAPE(input_shape[2],windows,stride,padding);
    return input_shape[0] * n_H * n_W;
}
std::vector<int64_t> dww::HEAverage2dLayer::out_shape(const std::vector<int64_t>& input_shape) const{
    assert(input_shape.size() == 3 && "The input's shape is not correct, it must be N * H * W!");
    int64_t n_H = SHAPE(input_shape[1],windows,stride,padding);
    int64_t n_W = SHAPE(input_shape[2],windows,stride,padding);
    return {input_shape[0],n_H,n_W};
}

std::ostream &dww::operator<<(std::ostream &out,const HEAverage2dLayer& self) {
    out << "\tWindow = " << self.windows << "; Padding = " << self.padding << "; Stride = " << self.stride << '\n';
    return out;
}

void dww::HESquare::forward(dww::Cipher_Tensor &sour, dww::HEWrapper &tools) {
    for(seal::Ciphertext& cipher : sour.value){
        tools.evl.square_inplace(cipher);
        tools.evl.relinearize_inplace(cipher,tools.rel_key);
        tools.evl.rescale_to_next_inplace(cipher);
    }
}
dww::HELinear::HELinear(std::vector<double>& w,std::vector<double>& b,int64_t in,int64_t out)
: weights(move(w)),bias(move(b)),in_(in),out_(out)
{

}
void dww::HELinear::forward(const dww::Cipher_Tensor &sour, dww::HEWrapper &tools, dww::Cipher_Tensor &dest) {
    assert(sour.numel() == in_ && "Matrix's col number is not equal to vector's length. Can't complete multiplication between matrix and vector!");
    assert(sour.shape.size() == 1 && dest.value.size() == out_ && "Linear's output size must be equal to Linear's out_");
    int64_t R = out_, C = in_; // 行数、列数
    // 权重矩阵与输入向量之间的乘积
    for(int64_t sz = 0; sz < R; ++sz){
        // 获取当前行的行向量元素
        std::vector<double> row_weight(weights.begin() + sz * C,weights.begin() + (sz + 1) * C);
        sour.dot_product_plain(row_weight,tools,dest.value[sz]);
    }
    // 加上偏置
    dest.add_plain_inplace(bias,tools);
}

int64_t dww::HELinear::out_numel() const{
    return out_;
}
std::vector<int64_t> dww::HELinear::out_shape() const{
    return {out_};
}

dww::HELinear::HELinear(const torch::Tensor &w, const torch::Tensor &b, int64_t in, int64_t out):
    weights(w.data_ptr<double>(),w.data_ptr<double>() + w.numel()),
    bias(w.data_ptr<double>(),w.data_ptr<double>() + w.numel()),
    in_(in),out_(out)
{

}

std::ostream &dww::operator<<(std::ostream &out,const HELinear& self) {
    out << "\tIn Channel = " << self.in_ << "; Out Channel = " << self.out_ << '\n';
    return out;
}



