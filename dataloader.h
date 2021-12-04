//
// Created by Dengweiwei on 2021/11/15.
//

#ifndef EXPERIMENT_DATALOADER_H
#define EXPERIMENT_DATALOADER_H
#include "datasets.h"
namespace dww{
    enum DATA_CAT{
        TRIN = 0,      //训练数据
        TEST,          //测试数据
        VAL            //valuation数据
    };

    struct BaseDataLoader{
        BaseDataLoader() = delete;
        explicit BaseDataLoader(int64_t batch) : batch_size(batch),samples_num(0){

        }
        virtual int64_t get_batch_num() = 0;
        // 所有 batch 的 image 样本
        std::vector<torch::Tensor> images;
        // 所有 batch 的 label 样本
        std::vector<torch::Tensor> labels;
        int64_t batch_size;
        int64_t samples_num;
    };

    struct MedDataSetLoader : BaseDataLoader{
        MedDataSetLoader() = delete;
        explicit MedDataSetLoader(const MedDataSet& data,DATA_CAT cat=TRIN,int64_t batch_sz = 32);
        int64_t get_batch_num() override;
    };

    struct MNISTDataSetLoader : BaseDataLoader{
        MNISTDataSetLoader() = delete;
        explicit MNISTDataSetLoader(const MNISTDataSet& data,DATA_CAT cat=TRIN,int64_t batch_sz = 32);
        int64_t get_batch_num() override;
    };
}

#endif //EXPERIMENT_DATALOADER_H
