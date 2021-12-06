//
// Created by Dengweiwei on 2021/11/19.
//
#include "he_utils.h"

#include <utility>
inline void dww::print_parameters(const seal::SEALContext &context)
{
    auto &context_data = *context.key_context_data();

    /*
    Which scheme are we using?
    */
    std::string scheme_name;
    switch (context_data.parms().scheme())
    {
        case seal::scheme_type::bfv:
            scheme_name = "BFV";
            break;
        case seal::scheme_type::ckks:
            scheme_name = "CKKS";
            break;
        default:
            throw std::invalid_argument("unsupported scheme");
    }
    std::cout << "/" << std::endl;
    std::cout << "| Encryption parameters :" << std::endl;
    std::cout << "|   scheme: " << scheme_name << std::endl;
    std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

    /*
    Print the size of the true (product) coefficient modulus.
    */
    std::cout << "|   coeff_modulus size: ";
    std::cout << context_data.total_coeff_modulus_bit_count() << " (";
    auto coeff_modulus = context_data.parms().coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++)
    {
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }
    std::cout << coeff_modulus.back().bit_count();
    std::cout << ") bits" << std::endl;

    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    if (context_data.parms().scheme() == seal::scheme_type::bfv)
    {
        std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
    }

    std::cout << "\\" << std::endl;
}

void dww::HEWrapper::get_plaintext_of(const std::vector<double>& input,seal::Plaintext& plain){
    encoder.encode(input,scale,plain);
}
void dww::HEWrapper::get_plaintext_of(double input, seal::Plaintext &plain) {
    encoder.encode(input,scale,plain);
}

std::vector<double> dww::HEWrapper::get_message_of(const seal::Plaintext &plain) {
    std::vector<double> result;
    encoder.decode(plain,result);
    return result;
}


void dww::HEWrapper::get_ciphertext_of(const seal::Plaintext& plain,seal::Ciphertext& dest) const{
    enc.encrypt(plain,dest);
}



std::vector<double> dww::HEWrapper::get_message_of(const seal::Ciphertext& cipher) {
    seal::Plaintext plain;
    dec.decrypt(cipher,plain);
    std::vector<double> res;
    encoder.decode(plain,res);
    return res;
}

void dww::HEWrapper::get_ciphertext_of(const std::vector<double> &input, seal::Ciphertext &dest) {
    seal::Plaintext plain;
    encoder.encode(input,scale,plain);
    enc.encrypt(plain,dest);
}


dww::Cipher_Tensor::Cipher_Tensor(int64_t len,const std::vector<int64_t>& s,int64_t bt) :value(len), shape(s), batch(bt){

}

dww::Cipher_Tensor::Cipher_Tensor(const torch::Tensor& image, HEWrapper& tools)
:value(image.numel() / image.size(0)), shape(image.sizes().begin()+1,image.sizes().end()),batch(image.size(0)){
    assert(image.size(0) <= tools.encoder.slot_count() && "The number of image can be batch encode is lager than slots!");
    assert(image.sizes().size() == 4 && "Tensor's shape must be N*C*H*W, where N is batch size, C is channel value, H is height of image, and W is width of image!");
    // 统一数据类型
    torch::Tensor image_copy = image.to(torch::kFloat64);

    // 获取 Tensor 中的所有像素值，并把其转为向量形式的 double 类型
    std::vector<double> pixel(image_copy.data_ptr<double>(),image_copy.data_ptr<double>() + image.numel());
    int64_t N = image_copy.size(0), C = image_copy.size(1), H = image_copy.size(2), W = image_copy.size(3);
    int64_t image_sz = C*H*W;
    for(int64_t c = 0; c < C; ++c){
        for(int64_t h = 0; h < H; ++h){
            for(int64_t w = 0; w < W; ++w){
                int64_t pos = c * H * W + h * W + w;
                std::vector<double> poly;
                for(int64_t n = 0; n < N; ++n) {
                    int64_t index = n * image_sz + pos;
                    poly.emplace_back(pixel[index]);
                }
                tools.get_ciphertext_of(poly,value[pos]);
            }
        }
    }
}
dww::Cipher_Tensor::Cipher_Tensor(const std::vector<seal::Ciphertext>& cipher,const std::vector<int64_t>& shape,int64_t bt)
:value(cipher),shape(shape),batch(bt)
{
}

void
dww::Cipher_Tensor::conv2d_one_window(int64_t channel, int64_t row,
                                      int64_t col,int64_t k,
                                      const std::vector<double>& weight,
                                      seal::Ciphertext& res,HEWrapper& tools) const {
    assert(channel < shape[0] && "The input value of channel is out range of Image's channel!");
    int64_t H = shape[1];
    int64_t W = shape[2];
    // channel 通道对应的起始下标
    int64_t start = channel * H * W;
    // 左上角行列下标
    std::pair<int,int> left_up(std::max(row,0L),std::max(col,0L));
    // 右下角行列下标
    std::pair<int,int> right_down(std::min(H-1,row+k-1),std::min(W-1,col+k-1));
    int64_t sz = (right_down.first - left_up.first + 1) * (right_down.second - left_up.second + 1);
    assert(sz == weight.size() && "The size of window in image is not equal to weights!");
    size_t id = 0;
    // 为了防止对 0 进行加密，这里直接对第一个像素值进行与权重的相乘，以避免加密 0 带来的精度损失
    seal::Plaintext temp1;
    tools.get_plaintext_of(weight[id++],temp1);
    int64_t index = start + W * left_up.first + left_up.second;
    if(temp1.parms_id() != value[index].parms_id())
        tools.evl.mod_switch_to_inplace(temp1,value[index].parms_id());
    tools.evl.multiply_plain(value[index],temp1,res);
    tools.evl.rescale_to_next_inplace(res);
    // 行
    for(int64_t i = left_up.first; i <= right_down.first; ++i){
        // 列
        for(int64_t j = left_up.second; j <= right_down.second; ++j){
            // 跳过第一个像素值
            if(i != left_up.first || j != left_up.second){
                // 获取下标值
                index = start + W * i + j;
                seal::Plaintext plain;
                tools.get_plaintext_of(weight[id++],plain);
                if(plain.parms_id() != value[index].parms_id())
                    tools.evl.mod_switch_to_inplace(plain,value[index].parms_id());
                seal::Ciphertext cipher;
                tools.evl.multiply_plain(value[index],plain,cipher);
                tools.evl.rescale_to_next_inplace(cipher);
                tools.evl.add_inplace(res,cipher);
            }
        }
    }
}

std::vector<std::vector<double>> dww::Cipher_Tensor::get_message_of_tensor(HEWrapper& tools) {
    std::vector<std::vector<double>> res(batch);
    for(seal::Ciphertext & cipher : value){
       std::vector<double> temp(move(tools.get_message_of(cipher)));
       for(int64_t i = 0; i < batch; ++i){
           res[i].emplace_back(temp[i]);
       }
    }
    return res;
}

void dww::Cipher_Tensor::average2d_one_window(int64_t channel,int64_t row,
                                              int64_t col, int64_t k,
                                              seal::Ciphertext &res,
                                              HEWrapper& tools) const {
    assert(channel < shape[0] && "The input value of channel is out range of Image's channel!");
    int64_t H = shape[1];
    int64_t W = shape[2];
    // channel 通道对应的起始下标
    int64_t start = channel * H * W;
    // 左上角行列下标
    std::pair<int,int> left_up(std::max(row,0L),std::max(col,0L));
    // 右下角行列下标
    std::pair<int,int> right_down(std::min(H-1,row+k-1),std::min(W-1,col+k-1));
    int64_t index = start + W * left_up.first + left_up.second;
    res = value[index];
    // 行
    for(int64_t i = left_up.first; i <= right_down.first; ++i){
        // 列
        for(int64_t j = left_up.second; j <= right_down.second; ++j){
            // 跳过第一个元素值
            if(i != left_up.first || j != left_up.second){
                // 获取下标值
                index = start + W * i + j;
                tools.evl.add_inplace(res,value[index]);
            }
        }
    }
    double temp = 1.0 / static_cast<double>(k*k);
    seal::Plaintext plain;
    tools.get_plaintext_of(temp,plain);
    tools.evl.mod_switch_to_inplace(plain,res.parms_id());
    tools.evl.multiply_plain_inplace(res,plain);
    tools.evl.rescale_to_next_inplace(res);
}

int64_t dww::Cipher_Tensor::numel() const {
    return value.size();
}

void dww::Cipher_Tensor::dot_product_plain(const std::vector<double> &vec,HEWrapper& tools, seal::Ciphertext &dest) const{
    assert(shape.size() == 1 && "*this object is not a vector, can not complete dot product!");
    assert(value.size() == vec.size() && "The two vector's length is not equal, can not complete dot product!");
    seal::Plaintext temp1;
    tools.get_plaintext_of(vec[0],temp1);
    tools.evl.mod_switch_to_inplace(temp1,value[0].parms_id());
    tools.evl.multiply_plain(value[0],temp1,dest);
    tools.evl.rescale_to_next_inplace(dest);
    for(int64_t sz = 1; sz < value.size(); ++sz) {
        seal::Plaintext temp2;
        seal::Ciphertext temp3;
        tools.get_plaintext_of(vec[sz],temp2);
        tools.evl.mod_switch_to_inplace(temp2, value[sz].parms_id());
        tools.evl.multiply_plain(value[sz], temp2, temp3);
        tools.evl.rescale_to_next_inplace(temp3);
        tools.evl.add_inplace(dest, temp3);
    }
}

void dww::Cipher_Tensor::add_plain_inplace(const std::vector<double> &vec, dww::HEWrapper &tools) {
    assert(shape.size() == 1 && value.size() == vec.size() && "Two vector's length is not equal, can not complete bit-wise addition!");
    for(int64_t sz = 0; sz < shape[0]; ++sz){
        seal::Plaintext temp;
        tools.get_plaintext_of(vec[sz],temp);
        tools.evl.mod_switch_to_inplace(temp,value[sz].parms_id());
        value[sz].scale() = temp.scale();
        tools.evl.add_plain_inplace(value[sz],temp);
    }
}


