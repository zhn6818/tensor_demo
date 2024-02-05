#include <iostream>
#include "superp.h"
#include "config.h"
#include "trtInterface.h"

int main(int argc, char** argv)
{
    std::cout << "hello world " << std::endl;
    Config config;
    config.onnx_file = "./superpoint/superpoint_v1.onnx";
    config.engine_file = "./superpoint/superpoint_v1.engine";
    config.input_tensor_names.push_back("input");

    std::shared_ptr<SuperPointC> superP = std::make_shared<SuperPointC>(&config);

    // superP.build(true);
    superP.get()->build();

    return 0;
}