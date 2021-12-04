//
// Created by Dengweiwei on 2021/11/15.
//

#include "dataloader.h"

dww::MedDataSetLoader::MedDataSetLoader(const MedDataSet& data,DATA_CAT cat,int64_t batch_sz):
        BaseDataLoader(batch_sz)
{
    torch::Tensor tensor_i,tensor_l;
    if(cat == TRIN){
        tensor_i = data.train_images;
        tensor_l = data.train_labels;
    }
    else if(cat == VAL){
        tensor_i = data.val_images;
        tensor_l = data.val_labels;
    }
    else if(cat == TEST){
        tensor_i = data.test_images;
        tensor_l = data.test_labels;
    }
    else
        assert(cat <= 2 && "MedDataSet don't exist this dataset!");
    assert(tensor_i.size(0) == tensor_l.size(0) && "The Number of Images don't equal to Labels!");
    samples_num = tensor_l.size(0);
    int64_t sz = 0;
    for(;sz + batch_sz < samples_num; sz += batch_sz){
        images.push_back(tensor_i.index({at::indexing::Slice(sz,sz+batch_sz)}));
        labels.push_back(tensor_l.index({at::indexing::Slice(sz,sz+batch_sz)}));
    }
    if(sz < samples_num){
        images.push_back(tensor_i.index({at::indexing::Slice(sz,samples_num)}));
        labels.push_back(tensor_l.index({at::indexing::Slice(sz,samples_num)}));
    }
}
int64_t dww::MedDataSetLoader::get_batch_num() {
    return static_cast<int64_t>(images.size());
}

dww::MNISTDataSetLoader::MNISTDataSetLoader(const MNISTDataSet& data,DATA_CAT cat,int64_t batch_sz): BaseDataLoader(batch_sz){
    torch::Tensor tensor_i,tensor_l;
    if(cat == TRIN){
        tensor_i = data.train_images;
        tensor_l = data.train_labels;
    }
    else if(cat == TEST){
        tensor_i = data.test_images;
        tensor_l = data.test_labels;
    }
    else
        assert(cat <= 1 && "MNISTDataSet don't exist this dataset!");
    assert(tensor_i.size(0) == tensor_l.size(0) && "The Number of Images don't equal to Labels!");
    samples_num = tensor_l.size(0);
    int64_t sz = 0;
    for(;sz + batch_sz < samples_num; sz += batch_sz){
        images.push_back(tensor_i.index({at::indexing::Slice(sz,sz+batch_sz)}));
        labels.push_back(tensor_l.index({at::indexing::Slice(sz,sz+batch_sz)}));
    }
    if(sz < samples_num){
        images.push_back(tensor_i.index({at::indexing::Slice(sz,samples_num)}));
        labels.push_back(tensor_l.index({at::indexing::Slice(sz,samples_num)}));
    }
}
int64_t dww::MNISTDataSetLoader::get_batch_num(){
    return static_cast<int64_t>(images.size());
}