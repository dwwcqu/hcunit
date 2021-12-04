//
// Created by Dengweiwei on 2021/11/19.
//

#ifndef EXPERIMENT_HE_UTILS_H
#define EXPERIMENT_HE_UTILS_H
#define PRECISION_DEGREE 10
#include <seal/seal.h>
#include <torch/torch.h>
namespace dww{
    inline void print_parameters(const seal::SEALContext &context);



    /**
     * 封装同态运算的参数:
     * EncryptionParameters
     */
    struct BaseHEWrapper{
        BaseHEWrapper(int64_t poly_m_d,double scale_bit) : params(seal::scheme_type::ckks),scale(pow(2,scale_bit)){
            // 设置多项式的阶
            params.set_poly_modulus_degree(poly_m_d);
            // 设置多项式系数的模空间大小，根据多项式的阶获取该阶对应的最大多项式模系数
            if(scale_bit == 25)
                params.set_coeff_modulus(seal::CoeffModulus::Create(poly_m_d,{45,25,25,25,25,25,45}));
            else if(scale_bit == 30)
                params.set_coeff_modulus(seal::CoeffModulus::Create(poly_m_d,
                                                                    {50,30,30,30,30,30,30,30,30,30,50}));
        }
        seal::EncryptionParameters params;
        double scale;
    };


    /**
     *  封装同态运算:
     *  SEALContext
     *  SecreteKey
     *  PublicKey
     *  RelinKeys
     *  CKKSEncoder
     */
    struct MidHEWrapper : BaseHEWrapper{
        MidHEWrapper(int64_t poly_m_d,double scale_bit):BaseHEWrapper(poly_m_d,scale_bit),
        context(params),
        gen(context),
        sec_key(gen.secret_key()),
        encoder(context)
        {
            gen.create_public_key(pub_key);
            gen.create_relin_keys(rel_key);
        }
        seal::SEALContext context;
        seal::KeyGenerator gen;
        seal::SecretKey sec_key;
        seal::PublicKey pub_key;
        seal::RelinKeys rel_key;
        seal::CKKSEncoder encoder;
    };



    /**
     *  封装同态运算：
     *  Encoder
     *  Decoder
     *  Evaluator
     */
    struct HEWrapper : MidHEWrapper{
        HEWrapper(int64_t poly_m_d,double scale_bit):MidHEWrapper(poly_m_d,scale_bit),
        enc(context,pub_key),
        dec(context,sec_key),
        evl(context)
        {}
        seal::Encryptor enc;
        seal::Decryptor dec;
        seal::Evaluator evl;

        /**
         *  把一个双精度的浮点数编码为所有 slots 都为 input 的明文多项式
         */
        void get_plaintext_of(const std::vector<double>& input,seal::Plaintext& plain);
        void get_plaintext_of(double input,seal::Plaintext& plain);
        int64_t get_slots_num() const{
            return encoder.slot_count();
        }
        /**
         * @param plain CKKS 明文多项式
         * @return 返回其明文对应的 double 类型的消息
         */
        std::vector<double> get_message_of(const seal::Plaintext& plain);
        std::vector<double> get_message_of(const seal::Ciphertext& cipher);
        /**
         * @param plain CKKS 明文多项式
         * @return 该明文多项式对应的密文
         */
        void get_ciphertext_of(const seal::Plaintext& plain,seal::Ciphertext& dest) const;
        void get_ciphertext_of(const std::vector<double>& input,seal::Ciphertext& dest);
    };

    class Cipher_Tensor{
    public:
        Cipher_Tensor() = delete;
        /**
         * @param len   代表图片像素值被 flatten 后的长度
         * @param s     代表图片像素值的形状
         *              构造一个 Cipher_Tensor 对象，但里面的内容都为空，需要对其进行赋值改变
         */
        explicit Cipher_Tensor(int64_t len,const std::vector<int64_t>& s,int64_t bt = 8192);
        /**
         * @param image     使用 torch::Tensor 表示的一个 batch 的 image 的像素值, 其 shape 为 N*C*H*W 的形式
         * @param tools     同态加密运算工具对象
         * 该构造函数的目的是根据 torch::Tensor 中保存的像素值，对其进行加密保存到对象 value 成员中
         * 需要注意：这里使用了 CKKSBatch 的批处理编码方法，可以把 tools.encoder.slots() 个图片的像素值保存到 C*H*W 个密文多项式中
         */
        explicit Cipher_Tensor(const torch::Tensor& image, HEWrapper& tools);
        Cipher_Tensor(const std::vector<seal::Ciphertext>& cipher,const std::vector<int64_t>& shape,int64_t bt = 8192);
        /**
         * @param channel   通道值
         * @param row       行号
         * @param col       列号
         * @param k         卷积核大小
         * @param weight    卷积权重
         * @param res       保存卷积结果
         * @param tools     同态运算工具
         *                  给定通道编号 channel, 像素值所处的下标 row, col, 和对应的卷积权重值，计算该密文态的卷积结果保存到 res 中
         */
        void conv2d_one_window(int64_t channel,
                               int64_t row,int64_t col,int64_t k,
                               const std::vector<double>& weight,seal::Ciphertext& res,
                               HEWrapper& tools) const;

        /**
         * @param channel   通道值
         * @param row       行号
         * @param col       列号
         * @param k         2D 平均池化结果
         * @param res       保存最终池化结果
         * @param tools     同态运算工具
         * 计算输入通道值为 channel, 下标为 row, col, 平均池化核大小为 k 的平均池化结果，并保存结果到 res 中
         */
        void average2d_one_window(int64_t channel,int64_t row,
                                  int64_t col,int64_t k,
                                  seal::Ciphertext& res,
                                  HEWrapper& tools) const;
        /**
         * @param vec       向量点乘的另一个明文向量
         * @param dest      保存点乘的最终结果
         * 一个密文向量乘上一个明文向量，并把结果保存在 dets 参数中
         */
        void dot_product_plain(const std::vector<double>& vec,HEWrapper& tools,seal::Ciphertext& dest) const;
        void add_plain_inplace(const std::vector<double>& vec,HEWrapper& tools);
        [[nodiscard]] int64_t numel() const;
        int64_t batch;
        /**
         * @param tools     对 Cipher_Tensor 加密的数据进行解密，并按照从 0 到 batch - 1 的下标返回每一个 image 的解密值，而每一个 image 的值保存在一个 vector 中，总共有 batch 个 vector
         * @return          返回 batch 个 image 图片的解密值
         */
        std::vector<std::vector<double>> get_message_of_tensor(HEWrapper& tools);
        std::vector<seal::Ciphertext> value;
        std::vector<int64_t> shape;
    };
}


#endif //EXPERIMENT_HE_UTILS_H
