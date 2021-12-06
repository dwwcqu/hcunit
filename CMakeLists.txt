cmake_minimum_required(VERSION 3.21)
project(Experiment)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# INCLUDE_DIRECTORIES(/home/dengweiwei/dwwlibs/include/)
INCLUDE_DIRECTORIES(/home/dengweiwei/dwwlibs/include/SEAL-3.7/)
INCLUDE_DIRECTORIES(/home/dengweiwei/libtorch/include/)
INCLUDE_DIRECTORIES(/home/dengweiwei/libtorch/include/torch/csrc/api/include/)

find_package(SEAL 3.7 REQUIRED)
find_package(Torch REQUIRED)
# find_package(OpenCV REQUIRED)

# include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(Experiment main.cpp cnpy.cpp cnpy.h train_model.h datasets.cpp datasets.h dataloader.cpp dataloader.h train_model.cpp he_layer.cpp he_layer.h he_utils.cpp he_utils.h conv_unit.h conv_unit.cpp lenet.cpp lenet.h CryptoNets.cpp CryptoNets.h LoLa_Dense.cpp LoLa_Dense.h)

target_link_libraries(Experiment SEAL::seal)
target_link_libraries(Experiment "${TORCH_LIBRARIES}")
# target_link_libraries(Experiment "${OpenCV_LIBS}")

set_property(TARGET Experiment PROPERTY CXX_STANDARD 17)