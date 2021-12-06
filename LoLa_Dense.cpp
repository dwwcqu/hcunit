//
// Created by 邓维维 on 2021/12/6.
//

#include "LoLa_Dense.h"
dww::LoLaDense::LoLaDense(const torch::Tensor &conv_w, const torch::Tensor &conv_b, int64_t conv_in, int64_t conv_out,
                          int64_t conv_k, int64_t conv_p, int64_t conv_s, const torch::Tensor &l1_w,
                          const torch::Tensor &l1_b, int64_t l1_in, int64_t l1_out, const torch::Tensor &l2_w,
                          const torch::Tensor &l2_b, int64_t l2_in, int64_t l2_out)
                          :
                          conv(conv_w,conv_b,conv_in,conv_out,conv_k,conv_p,conv_s),
                          linear1(l1_w,l1_b,l1_in,l1_out),
                          linear2(l2_w,l2_b,l2_in,l2_out)
{
}

void dww::LoLaDense::forward(const torch::Tensor &input, dww::HEWrapper &tools, dww::Cipher_Tensor &output) {
    assert(input.sizes().size() == 4 && "The input image is not a 2D image, its shape must be N * C * H * W!");
    std::chrono::high_resolution_clock::time_point start,end;
    start = std::chrono::high_resolution_clock::now();
    Cipher_Tensor input_cipher(input,tools);
    end = std::chrono::high_resolution_clock::now();
    enc_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    // 获取当前同态运算的批处理的 image 图片数量，以用于后面的 Cipher_Tensor 中的 batch 的设置
    int64_t batch = input_cipher.batch;
    Cipher_Tensor conv_output(conv.out_numel(input_cipher.shape),conv.out_shape(input_cipher.shape),batch);
    Cipher_Tensor linear1_output(linear1.out_,{linear1.out_},batch);

    start = std::chrono::high_resolution_clock::now();
    conv.forward(input_cipher,tools,conv_output);
    end = std::chrono::high_resolution_clock::now();
    conv_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    square.forward(conv_output,tools);
    end = std::chrono::high_resolution_clock::now();
    square_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    conv_output.shape = {conv_output.numel()};
    linear1.forward(conv_output,tools,linear1_output);
    end = std::chrono::high_resolution_clock::now();
    linear_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    square.forward(linear1_output,tools);
    end = std::chrono::high_resolution_clock::now();
    square_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    linear2.forward(linear1_output,tools,output);
    end = std::chrono::high_resolution_clock::now();
    linear_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}
void dww::LoLaDense_he_inference_test(const std::string& filename,int64_t poly_d,int64_t scale){
    std::string path = "../model/loladense/" + filename + "/";
    torch::Tensor conv_weight,conv_bias;
    torch::Tensor linear1_weight,linear1_bias;
    torch::Tensor linear2_weight,linear2_bias;

    torch::load(conv_weight,path + "conv_weight.pt");
    conv_weight = conv_weight.to(c10::kCPU);
    torch::load(conv_bias,path + "conv_bias.pt");
    conv_bias = conv_bias.to(c10::kCPU);

    torch::load(linear1_weight,path + "linear1_weight.pt");
    linear1_weight = linear1_weight.to(c10::kCPU);
    torch::load(linear1_bias,path + "linear1_bias.pt");
    linear1_bias = linear1_bias.to(c10::kCPU);

    torch::load(linear2_weight,path + "linear2_weight.pt");
    linear2_weight = linear2_weight.to(c10::kCPU);
    torch::load(linear2_bias,path + "linear2_bias.pt");
    linear2_bias = linear2_bias.to(c10::kCPU);

    int64_t conv1_in = 1,conv1_out = 5, conv1_k = 5, conv1_s = 2,conv1_p = 2;
    int64_t l1_in = 5*14*14,l1_out = 100;
    int64_t l2_in = 100,l2_out = dww::label_num.at(filename);

    LoLaDense model(conv_weight,conv_bias,conv1_in,conv1_out,conv1_k,conv1_p,conv1_s,
                     linear1_weight,linear1_bias,
                     l1_in,l1_out,
                     linear2_weight,linear2_bias,
                     l2_in,l2_out
    );
    HEWrapper tools(poly_d,scale);
    dww::MedDataSet dataset(filename);
    dww::MedDataSetLoader dataloader(dataset,DATA_CAT::TEST,tools.get_slots_num());
    int64_t sz = dataloader.get_batch_num();
    int64_t samples_num = dataloader.samples_num;
    int64_t correct = 0;
    std::ofstream test_log("../experiment/loladense",std::ios_base::app);
    assert(test_log.is_open() && "File loladense can not open!");
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
std::ostream& dww::operator<<(std::ostream& out,const LoLaDense& self){
    out << "conv: " << self.conv
    << "linear1: " << self.linear1
    << "linear2: " << self.linear2;
    return out;
}