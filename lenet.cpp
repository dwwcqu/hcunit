//
// Created by Dengweiwei on 2021/12/3.
//
#include "lenet.h"
[[maybe_unused]] dww::HELeNet_Model::HELeNet_Model(std::vector<double> &conv1_w, std::vector<double> &conv1_b, int64_t conv1_in,
                                                   int64_t conv1_out, int64_t conv1_k, int64_t conv1_p, int64_t conv1_s, int64_t pool1_k,
                                                   int64_t pool1_p, int64_t pool1_s, std::vector<double> &conv2_w,
                                                   std::vector<double> &conv2_b, int64_t conv2_in, int64_t conv2_out, int64_t conv2_k,
                                                   int64_t conv2_p, int64_t conv2_s, int64_t pool2_k, int64_t pool2_p, int64_t pool2_s,
                                                   std::vector<double> &conv3_w, std::vector<double> &conv3_b, int64_t conv3_in,
                                                   int64_t conv3_out, int64_t conv3_k, int64_t conv3_p, int64_t conv3_s,
                                                   std::vector<double> &l1_w, std::vector<double> &l1_b, int64_t l1_in, int64_t l1_out,
                                                   std::vector<double> &l2_w, std::vector<double> &l2_b, int64_t l2_in, int64_t l2_out)
        :
        conv1(conv1_w,conv1_b,conv1_in,conv1_out,conv1_k,conv1_p,conv1_s),
        pool1(pool1_k,pool1_p,pool1_s),
        conv2(conv2_w,conv2_b,conv2_in,conv2_out,conv2_k,conv2_p,conv2_s),
        pool2(pool2_k,pool2_p,pool2_s),
        conv3(conv3_w,conv3_b,conv3_in,conv3_out,conv3_k,conv3_p,conv3_s),
        linear1(l1_w,l1_b,l1_in,l1_out),
        linear2(l2_w,l2_b,l2_in,l2_out)
{
}
dww::HELeNet_Model::HELeNet_Model(const torch::Tensor &conv1_w, const torch::Tensor &conv1_b, int64_t conv1_in,
                                  int64_t conv1_out, int64_t conv1_k, int64_t conv1_p, int64_t conv1_s, int64_t pool1_k,
                                  int64_t pool1_p, int64_t pool1_s, const torch::Tensor &conv2_w,
                                  const torch::Tensor &conv2_b, int64_t conv2_in, int64_t conv2_out, int64_t conv2_k,
                                  int64_t conv2_p, int64_t conv2_s, int64_t pool2_k, int64_t pool2_p, int64_t pool2_s,
                                  const torch::Tensor &conv3_w, const torch::Tensor &conv3_b, int64_t conv3_in,
                                  int64_t conv3_out, int64_t conv3_k, int64_t conv3_p, int64_t conv3_s,
                                  const torch::Tensor &l1_w, const torch::Tensor &l1_b, int64_t l1_in, int64_t l1_out,
                                  const torch::Tensor &l2_w, const torch::Tensor &l2_b, int64_t l2_in, int64_t l2_out)
        :
        conv1(conv1_w,conv1_b,conv1_in,conv1_out,conv1_k,conv1_p,conv1_s),
        pool1(pool1_k,pool1_p,pool1_s),
        conv2(conv2_w,conv2_b,conv2_in,conv2_out,conv2_k,conv2_p,conv2_s),
        pool2(pool2_k,pool2_p,pool2_s),
        conv3(conv3_w,conv3_b,conv3_in,conv3_out,conv3_k,conv3_p,conv3_s),
        linear1(l1_w,l1_b,l1_in,l1_out),
        linear2(l2_w,l2_b,l2_in,l2_out)
{

}
void dww::HELeNet_Model::forward(const torch::Tensor &input, dww::HEWrapper &tools, dww::Cipher_Tensor &output) {
    assert(input.sizes().size() == 4 && "The input image is not a 2D image, its shape must be N * C * H * W!");
    std::chrono::high_resolution_clock::time_point start,end;
    start = std::chrono::high_resolution_clock::now();
    Cipher_Tensor input_cipher(input,tools);
    end = std::chrono::high_resolution_clock::now();
    enc_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    // 获取当前同态运算的批处理的 image 图片数量，以用于后面的 Cipher_Tensor 中的 batch 的设置
    int64_t batch = input_cipher.batch;
    Cipher_Tensor conv1_output(conv1.out_numel(input_cipher.shape),conv1.out_shape(input_cipher.shape),batch);
    Cipher_Tensor pool1_output(pool1.out_numel(conv1_output.shape),pool1.out_shape(conv1_output.shape),batch);
    Cipher_Tensor conv2_output(conv2.out_numel(pool1_output.shape),conv2.out_shape(pool1_output.shape),batch);
    Cipher_Tensor pool2_output(pool2.out_numel(conv2_output.shape),pool2.out_shape(conv2_output.shape),batch);
    Cipher_Tensor conv3_output(conv3.out_numel(pool2_output.shape),conv3.out_shape(pool2_output.shape),batch);
    Cipher_Tensor linear1_output(linear1.out_,{linear1.out_},batch);



    start = std::chrono::high_resolution_clock::now();
    conv1.forward(input_cipher,tools,conv1_output);
    end = std::chrono::high_resolution_clock::now();
    conv_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();


    start = std::chrono::high_resolution_clock::now();
    pool1.forward(conv1_output,tools,pool1_output);
    end = std::chrono::high_resolution_clock::now();
    pool_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();


    start = std::chrono::high_resolution_clock::now();
    square.forward(pool1_output,tools);
    end = std::chrono::high_resolution_clock::now();
    square_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();


    start = std::chrono::high_resolution_clock::now();
    conv2.forward(pool1_output,tools,conv2_output);
    end = std::chrono::high_resolution_clock::now();
    conv_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();


    start = std::chrono::high_resolution_clock::now();
    pool2.forward(conv2_output,tools,pool2_output);
    end = std::chrono::high_resolution_clock::now();
    pool_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    square.forward(pool2_output,tools);
    end = std::chrono::high_resolution_clock::now();
    square_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    conv3.forward(pool2_output,tools,conv3_output);
    end = std::chrono::high_resolution_clock::now();
    conv_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();


    start = std::chrono::high_resolution_clock::now();
    conv3_output.shape = {conv3_output.numel()};
    linear1.forward(conv3_output,tools,linear1_output);
    end = std::chrono::high_resolution_clock::now();
    linear_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();


    start = std::chrono::high_resolution_clock::now();
    linear2.forward(linear1_output,tools,output);
    end = std::chrono::high_resolution_clock::now();
    linear_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}



std::ostream &dww::operator<<(std::ostream &out,const HELeNet_Model& self){
    out << "conv1: " << self.conv1 << "pool1: " << self.pool1
    << "conv2: " << self.conv2 << "pool2: " << self.pool2
    << "conv3: " << self.conv3 << "linear1: " << self.linear1
    << "linear2: " << self.linear2;
    return out;
}

void dww::LeNet_he_inference_test(const std::string& filename){
    std::string path = "../model/lenet/" + filename + "/";
    torch::Tensor conv1_weight,conv1_bias;
    torch::Tensor conv2_weight,conv2_bias;
    torch::Tensor conv3_weight,conv3_bias;
    torch::Tensor linear1_weight,linear1_bias;
    torch::Tensor linear2_weight,linear2_bias;

    torch::load(conv1_weight,path + "conv1_weight.pt");
    conv1_weight = conv1_weight.to(c10::kCPU);
    torch::load(conv1_bias,path + "conv1_bias.pt");
    conv1_bias = conv1_bias.to(c10::kCPU);

    torch::load(conv2_weight,path + "conv2_weight.pt");
    conv2_weight = conv2_weight.to(c10::kCPU);
    torch::load(conv2_bias,path + "conv2_bias.pt");
    conv2_bias = conv2_bias.to(c10::kCPU);

    torch::load(conv3_weight,path + "conv3_weight.pt");
    conv3_weight = conv3_weight.to(c10::kCPU);
    torch::load(conv3_bias,path + "conv3_bias.pt");
    conv3_bias = conv3_bias.to(c10::kCPU);

    torch::load(linear1_weight,path + "linear1_weight.pt");
    linear1_weight = linear1_weight.to(c10::kCPU);
    torch::load(linear1_bias,path + "linear1_bias.pt");
    linear1_bias = linear1_bias.to(c10::kCPU);

    torch::load(linear2_weight,path + "linear2_weight.pt");
    linear2_weight = linear2_weight.to(c10::kCPU);
    torch::load(linear2_bias,path + "linear2_bias.pt");
    linear2_bias = linear2_bias.to(c10::kCPU);

    int64_t conv1_in = 1,conv1_out = 6, conv1_k = 3, conv1_s = 1,conv1_p = 0;
    int64_t conv2_in = 6,conv2_out = 16, conv2_k = 3, conv2_s = 1,conv2_p = 0;
    int64_t conv3_in = 16,conv3_out = 120, conv3_k = 3, conv3_s = 1,conv3_p = 0;
    int64_t pool1_k = 2,pool1_s = 2,pool1_p = 0;
    int64_t pool2_k = 2,pool2_s = 2,pool2_p = 0;
    int64_t l1_in = 120*3*3,l1_out = 84;
    int64_t l2_in = 84,l2_out = dww::label_num.at(filename);

    HELeNet_Model model(conv1_weight,conv1_bias,conv1_in,conv1_out,conv1_k,conv1_p,conv1_s,
                        pool1_k,pool1_p,pool1_s,
                        conv2_weight,conv2_bias,conv2_in,conv2_out,conv2_k,conv2_p,conv2_s,
                        pool2_k,pool2_p,pool2_s,
                        conv3_weight,conv3_bias,conv3_in,conv3_out,conv3_k,conv3_p,conv3_s,
                        linear1_weight,linear1_bias,
                        l1_in,l1_out,
                        linear2_weight,linear2_bias,
                        l2_in,l2_out
    );
    HEWrapper tools(16384,30);
    dww::MedDataSet dataset(filename);
    dww::MedDataSetLoader dataloader(dataset,DATA_CAT::TEST,tools.get_slots_num());
    int64_t sz = dataloader.get_batch_num();
    int64_t samples_num = dataloader.samples_num;
    int64_t correct = 0;
    std::ofstream test_log("../experiment/lenet",std::ios_base::app);
    assert(test_log.is_open() && "File lenet can not open!");
    test_log << "Dataset: " << filename << '\n';
    test_log << "Model Information: \n";
    test_log << model;
    std::cout << "----> Homomorphic Convolution " <<  filename <<  " Datasets Starts <-----\n";
    using std::chrono::high_resolution_clock;
    high_resolution_clock::time_point start,end;
    double time_consume = 0;
    for(int64_t i = 0; i < sz; ++i){
        torch::Tensor image = dataloader.images[i];
        torch::Tensor label = dataloader.labels[i];
        // 该 batch 中有的 image 图片个数
        int64_t bt_sz = image.size(0);
        // 保存最终预测结果的密文值
        Cipher_Tensor output(model.linear2.out_,{model.linear2.out_},bt_sz);
        start = high_resolution_clock::now();
        // 同态运算一个卷积单元
        model.forward(image,tools,output);
        end = high_resolution_clock::now();
        time_consume += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

        start = high_resolution_clock::now();
        // 对预测结果进行解密，获得每一个 image 的预测结果
        std::vector<std::vector<double>> res(output.get_message_of_tensor(tools));
        end = high_resolution_clock::now();
        model.dec_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        // 记录每一个 image 的预测结果，使用下标的方式反映(某一个下标处值最大便是保存该下标值)
        std::vector<int64_t> max_index(bt_sz);
        for(int64_t j = 0; j < bt_sz; ++j)
            max_index[j] = std::max_element(res[j].begin(),res[j].end()) - res[j].begin();
        torch::Tensor temp = torch::tensor(max_index,torch::TensorOptions(torch::kInt64));
        assert(temp.numel() == label.numel() && "The prediction number is not equal to labels!");
        correct += (bt_sz - (temp - label).count_nonzero().item<int64_t>());
        std::cout << "Dataset " << filename << ",batch " << i + 1 << " completed!\n";
        test_log << "\t[" << i + 1 << "/" << sz << "], Acc = " << static_cast<double>(correct) / samples_num * 100 << "%\n";
    }

    test_log << "Total Inference Images                             : " << dataloader.samples_num << "\n";
    test_log << "Total Acc                                          : " << static_cast<double>(correct) / samples_num * 100 << "%\n";
    test_log << "Total Convolution Operation Time Consume           : " << model.conv_time << "(s)\n";
    test_log << "Average Convolution Operation Time Consume         : " << model.conv_time / samples_num << "(s)\n";
    test_log << "Total Pooling Operation Time Consume               : " << model.pool_time << "(s)\n";
    test_log << "Average Pooling Operation Time Consume             : " << model.pool_time / samples_num << "(s)\n";
    test_log << "Total Square Activation Operation Time Consume     : " << model.square_time << "(s)\n";
    test_log << "Average Square Activation Operation Time Consume   : " << model.square_time / samples_num << "(s)\n";
    test_log << "Total Linear Operation Time Consume                : " << model.linear_time << "(s)\n";
    test_log << "Average Square Activation Operation Time Consume   : " << model.linear_time / samples_num << "(s)\n";
    test_log << "Total Time Consume                                 : " << time_consume << "(s)\n";
    test_log << "Average Time Consume Per Batch                     : " << time_consume / sz << "(s)\n";
    test_log << "Average Time Consume Per Image                     : " << time_consume / samples_num << "(s)\n\n";
    test_log.flush();
    test_log.close();
    std::cout << "----> Homomorphic Convolution " <<  filename <<  " Datasets End <-----\n";
}