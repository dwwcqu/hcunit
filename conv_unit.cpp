//
// Created by Dengweiwi on 2021/12/3.
//

#include "conv_unit.h"

dww::HETest_1_1_CHAN_Model::HETest_1_1_CHAN_Model(
        std::vector<double>& conv2d_w,std::vector<double>& conv2d_b,
        int64_t conv2d_in,int64_t conv2d_out,int64_t conv2d_k,int64_t conv2d_p,int64_t conv2d_s,
        int64_t pool2d_k,int64_t pool2d_p,int64_t pool2d_s,
        std::vector<double>& l1_w, std::vector<double>& l1_b,
        int64_t l1_in,int64_t l1_out,
        std::vector<double>& l2_w,std::vector<double>& l2_b,
        int64_t l2_in,int64_t l2_out)
        :
        conv2d(conv2d_w,conv2d_b,conv2d_in,conv2d_out,conv2d_k,conv2d_p,conv2d_s),
        pool2d(pool2d_k,pool2d_p,pool2d_s),
        linear1(l1_w,l1_b,l1_in,l1_out),
        linear2(l2_w,l2_b,l2_in,l2_out)
{

}

void dww::HETest_1_1_CHAN_Model::forward(const torch::Tensor &input, dww::HEWrapper &tools, dww::Cipher_Tensor &output) {
    assert(input.sizes().size() == 4 && "The input image is not a 2D image, its shape must be N * C * H * W!");
    std::chrono::high_resolution_clock::time_point start,end;
    start = std::chrono::high_resolution_clock::now();
    Cipher_Tensor conv2d_input(input,tools);
    end = std::chrono::high_resolution_clock::now();
    time_enc += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    // 获取当前同态运算的批处理的 image 图片数量，以用于后面的 Cipher_Tensor 中的 batch 的设置
    int64_t batch = conv2d_input.batch;
    Cipher_Tensor conv2d_output(conv2d.out_numel(conv2d_input.shape),conv2d.out_shape(conv2d_input.shape),batch);
    Cipher_Tensor pool2d_output(pool2d.out_numel(conv2d_output.shape),pool2d.out_shape(conv2d_output.shape),batch);
    Cipher_Tensor linear1_output(linear1.out_,{linear1.out_},batch);

    start = std::chrono::high_resolution_clock::now();
    conv2d.forward(conv2d_input,tools,conv2d_output);
    end = std::chrono::high_resolution_clock::now();
    time_conv += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    pool2d.forward(conv2d_output,tools,pool2d_output);
    end = std::chrono::high_resolution_clock::now();
    time_pool += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    square.forward(pool2d_output,tools);
    end = std::chrono::high_resolution_clock::now();
    time_relu += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    pool2d_output.shape = {pool2d_output.numel()};
    linear1.forward(pool2d_output,tools,linear1_output);
    end = std::chrono::high_resolution_clock::now();
    time_l1 += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();


    start = std::chrono::high_resolution_clock::now();
    linear2.forward(linear1_output,tools,output);
    end = std::chrono::high_resolution_clock::now();
    time_l2 += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

void dww::conv_unit_he_inference_test(const std::string& filename,int64_t poly_d,int64_t scale){
    std::string path = "../model/conv_unit/" + filename + "/";
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

    int64_t conv_in = 1,conv_out = 8, conv_k = 3, conv_s = 1,conv_p = 1;
    int64_t pool_k = 2,pool_s = 2,pool_p = 1;
    int64_t l1_in = 8*15*15,l1_out = 64;
    int64_t l2_in = 64,l2_out = dww::label_num.at(filename);
    std::vector<double> conv_w(conv_weight.data_ptr<double>(),conv_weight.data_ptr<double>() + conv_weight.numel());
    std::vector<double> conv_b(conv_bias.data_ptr<double>(),conv_bias.data_ptr<double>() + conv_bias.numel());

    std::vector<double> linear1_w(linear1_weight.data_ptr<double>(),linear1_weight.data_ptr<double>() + linear1_weight.numel());
    std::vector<double> linear1_b(linear1_bias.data_ptr<double>(),linear1_bias.data_ptr<double>() + linear1_bias.numel());


    std::vector<double> linear2_w(linear2_weight.data_ptr<double>(),linear2_weight.data_ptr<double>() + linear2_weight.numel());
    std::vector<double> linear2_b(linear2_bias.data_ptr<double>(),linear2_bias.data_ptr<double>() + linear2_bias.numel());
    HETest_1_1_CHAN_Model model(conv_w,conv_b,
                                conv_in,conv_out,conv_k,conv_p,conv_s,
                                pool_k,pool_p,pool_s,
                                linear1_w,linear1_b,
                                l1_in,l1_out,
                                linear2_w,linear2_b,
                                l2_in,l2_out
    );
    HEWrapper tools(poly_d,scale);
    dww::MedDataSet dataset(filename);
    dww::MedDataSetLoader dataloader(dataset,DATA_CAT::TEST,tools.get_slots_num());
    int64_t sz = dataloader.get_batch_num();
    int64_t samples_num = dataloader.samples_num;
    int64_t correct = 0;
    std::ofstream test_log("../experiment/conv_unit",std::ios_base::app);
    assert(test_log.is_open() && "File conv_unit can not open!");
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
        model.time_enc += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        start = high_resolution_clock::now();
        // 同态运算一个卷积单元
        model.forward(image,tools,output);
        end = high_resolution_clock::now();
        time_consume += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

        start = high_resolution_clock::now();
        // 对预测结果进行解密，获得每一个 image 的预测结果
        std::vector<std::vector<double>> res(output.get_message_of_tensor(tools));
        end = high_resolution_clock::now();
        model.time_dec += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
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

    test_log << "Total Inference Images                     : " << dataloader.samples_num << "\n";
    test_log << "Total Acc                                  : " << static_cast<double>(correct) / samples_num * 100 << "%\n";
    test_log << "Total Conv2d Operation Time Consume        : " << model.time_conv << "(s)\n";
    test_log << "Average Conv2d Operation Time Consume      : " << model.time_conv / samples_num << "(s)\n";
    test_log << "Total Average2d Operation Time Consume     : " << model.time_pool << "(s)\n";
    test_log << "Average Average2d Operation Time Consume   : " << model.time_pool / samples_num << "(s)\n";
    test_log << "Total LeakyReLU Operation Time Consume     : " << model.time_relu << "(s)\n";
    test_log << "Average LeakyReLU Operation Time Consume   : " << model.time_relu / samples_num << "(s)\n";
    test_log << "Total Linear1 Operation Time Consume       : " << model.time_l1 << "(s)\n";
    test_log << "Average Linear1 Operation Time Consume     : " << model.time_l1 / samples_num << "(s)\n";
    test_log << "Total Linear2 Operation Time Consume       : " << model.time_l2 << "(s)\n";
    test_log << "Average Linear2 Operation Time Consume     : " << model.time_l2 / samples_num << "(s)\n";
    test_log << "Total Encryption Time Consume              : " << model.time_enc << "(s)\n";
    test_log << "Average Encryption Time Consume Per Images : " << model.time_enc / samples_num << "(s)\n";
    test_log << "Total Decryption Time Consume              : " << model.time_dec << "(s)\n";
    test_log << "Average Decryption Time Consume Per Images : " << model.time_dec / samples_num << "(s)\n";
    test_log << "Total Time Consume                         : " << time_consume << "(s)\n";
    test_log << "Average Time Consume Per Batch             : " << time_consume / sz << "(s)\n";
    test_log << "Average Time Consume Per Image             : " << time_consume / samples_num << "(s)\n\n";
    test_log.flush();
    test_log.close();
    std::cout << "----> Homomorphic Convolution " <<  filename <<  " Datasets End <-----\n";
}

std::ostream& dww::operator<<(std::ostream& out,const HETest_1_1_CHAN_Model& self){
    out << "conv: " << self.conv2d
    << "pool: " << self.pool2d
    << "linear1: " << self.linear1
    << "linear2: " << self.linear2;
    return out;
}