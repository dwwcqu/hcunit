//
// Created by 邓维维 on 2021/11/13.
//

#ifndef EXPERIMENT_DATASETS_H
#define EXPERIMENT_DATASETS_H
#include <string>
#include <vector>
#include <map>
#include "cnpy.h"
#include <torch/torch.h>
#define ADD_MED_SUFFIX(filename) "../medmnist/"+ filename+".npz"
#define ADD_MNIST_SUFFIX(filename) "../" + filename + ".npz"
namespace dww{
    using namespace std;
    static const map<string,int64_t> channel_num{
            {"pathmnist",3},
            {"chestmnist",1},
            {"dermamnist",3},
            {"octmnist",1},
            {"pneumoniamnist",1},
            {"retinamnist",3},
            {"breastmnist",1},
            {"bloodmnist",3},
            {"tissuemnist",1},
            {"organamnist",1},
            {"organcmnist",1},
            {"organsmnist",1}
    };
    static const map<string,int64_t> label_num{
            {"pathmnist",9},
            {"chestmnist",14},
            {"dermamnist",7},
            {"octmnist",4},
            {"pneumoniamnist",2},
            {"retinamnist",5},
            {"breastmnist",2},
            {"bloodmnist",8},
            {"tissuemnist",8},
            {"organamnist",11},
            {"organcmnist",11},
            {"organsmnist",11}
    };
    static const map<string,string> task{
            {"pathmnist","multi-class"},
            {"chestmnist","multi-label, binary-class"},
            {"dermamnist","multi-class"},
            {"octmnist","multi-class"},
            {"pneumoniamnist","binary-class"},
            {"retinamnist","ordinal-regression"},
            {"breastmnist","binary-class"},
            {"bloodmnist","multi-class"},
            {"tissuemnist","multi-class"},
            {"organamnist","multi-class"},
            {"organcmnist","multi-class"},
            {"organsmnist","multi-class"}
    };
    static const map<string,map<size_t,string>> labels_info
            {
                    {"pathmnist",
                     {
                             {0, "adipose"},
                             {1, "background"},
                             {2, "debris"},
                             {3, "lymphocytes"},
                             {4, "mucus"},
                             {5, "smooth muscle"},
                             {6, "normal colon mucosa"},
                             {7, "cancer-associated stroma"},
                             {8, "colorectal adenocarcinoma epithelium"}
                     }
                    },
                    {"chestmnist",
                     {
                             {0,"atelectasis"},
                             {1,"cardiomegaly"},
                             {2,"effusion"},
                             {3,"infiltration"},
                             {4,"mass"},
                             {5,"nodule"},
                             {6,"pneumonia"},
                             {7,"pneumothorax"},
                             {8,"consolidation"},
                             {9,"edema"},
                             {10,"emphysema"},
                             {11,"fibrosis"},
                             {12,"pleural"},
                             {13,"hernia"}
                     }
                     },
                    {"dermamnist",
                     {
                             {0,"actinic keratoses and intraepithelial carcinoma"},
                             {1,"basal cell carcinoma"},
                             {2,"benign keratosis-like lesions"},
                             {3,"dermatofibroma"},
                             {4,"melanoma"},
                             {5,"melanocytic nevi"},
                             {6,"vascular lesions"}
                     }
                     },
                    {"octmnist",
                     {
                             {0,"choroidal neovascularization"},
                             {1,"diabetic macular edema"},
                             {2,"drusen"},
                             {3,"normal"}
                     }
                     },
                    {"pneumoniamnist",
                     {
                             {0,"normal"},
                             {1,"pneumonia"}
                     }
                     },
                    {"retinamnist",
                     {
                             {0,"0"},
                             {1,"1"},
                             {2,"2"},
                             {3,"3"},
                             {4,"4"}
                     }
                    },
                    {"breastmnist",
                     {
                             {0,"malignant"},
                             {1,"normal, benign"}
                     }
                     },
                    {"bloodmnist",
                     {
                             {0,"basophil"},
                             {1,"eosinophil"},
                             {2,"erythroblast"},
                             {3,"ig"},
                             {4,"lymphocyte"},
                             {5,"monocyte"},
                             {6,"neutrophil"},
                             {7,"platelet"}
                     }
                     },
                    {"tissuemnist",
                     {
                             {0,"Collecting Duct, Connecting Tubule"},
                             {1,"Distal Convoluted Tubule"},
                             {2,"Glomerular endothelial cells"},
                             {3,"Interstitial endothelial cells"},
                             {4,"Leukocytes"},
                             {5,"Podocytes"},
                             {6,"Proximal Tubule Segments"},
                             {7,"Thick Ascending Limb"}
                     }
                    },
                    {"organamnist",
                     {
                             {0,"bladder"},
                             {1,"femur-left"},
                             {2,"femur-right"},
                             {3,"heart"},
                             {4,"kidney-left"},
                             {5,"kidney-right"},
                             {6,"liver"},
                             {7,"lung-left"},
                             {8,"lung-right"},
                             {9,"pancreas"},
                             {10,"spleen"}
                     }
                    },
                    {"organcmnist",
                     {
                            {0,"bladder"},
                            {1,"femur-left"},
                            {2,"femur-right"},
                            {3,"heart"},
                            {4,"kidney-left"},
                            {5,"kidney-right"},
                            {6,"liver"},
                            {7,"lung-left"},
                            {8,"lung-right"},
                            {9,"pancreas"},
                            {10,"spleen"}
                     }
                    },
                    {"organsmnist",
                     {
                             {0,"bladder"},
                             {1,"femur-left"},
                             {2,"femur-right"},
                             {3,"heart"},
                             {4,"kidney-left"},
                             {5,"kidney-right"},
                             {6,"liver"},
                             {7,"lung-left"},
                             {8,"lung-right"},
                             {9,"pancreas"},
                             {10,"spleen"}
                     }
                     }
            };
    static const map<string,map<string,int64_t>> samples
    {
            {"pathmnist",{{"train",89996},{"val",10004},{"test",7180}}},
            {"chestmnist",{{"train",78468},{"val",11219},{"test",22433}}},
            {"dermamnist",{{"train",7007},{"val",1003},{"test",2005}}},
            {"octmnist",{{"train",97477},{"val",10832},{"test",1000}}},
            {"pneumoniamnist",{{"train",4708},{"val",524},{"test",624}}},
            {"retinamnist",{{"train",1080},{"val",120},{"test",400}}},
            {"breastmnist",{{"train",546},{"val",78},{"test",156}}},
            {"bloodmnist",{{"train",11959},{"val",1712},{"test",3421}}},
            {"tissuemnist",{{"train",165466},{"val",23640},{"test",47280}}},
            {"organamnist",{{"train",34581},{"val",6491},{"test",17778}}},
            {"organcmnist",{{"train",13000},{"val",2392},{"test",8268}}},
            {"organsmnist",{{"train",13940},{"val",2452},{"test",8829}}}
    };

    void print_all_dataset_info();
    void print_all_dataset_samples_info();
    torch::Tensor get_images_tensor_from_NpyArray(cnpy::NpyArray& npy);
    torch::Tensor get_labels_tensor_from_NpyArray(cnpy::NpyArray& npy);
    struct BaseDataSet{
        BaseDataSet() = default;
        torch::Tensor train_images,train_labels;
        torch::Tensor test_images,test_labels;
    };
    struct MedDataSet : BaseDataSet{
        MedDataSet() = delete;
        explicit MedDataSet(const std::string& filename);

        torch::Tensor val_images,val_labels;
    };

    struct MNISTDataSet : BaseDataSet{
        MNISTDataSet() = delete;
        explicit MNISTDataSet(const std::string& filename);
    };
}
#endif //EXPERIMENT_DATASETS_H
