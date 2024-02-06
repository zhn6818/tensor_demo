#include <iostream>
#include "superp.h"
#include "config.h"
#include "trtInterface.h"
#include <opencv2/opencv.hpp>


int main(int argc, char** argv)
{
    std::cout << "hello world " << std::endl;
    Config config;
    config.onnx_file = "./superpoint/superpoint_v1.onnx";
    config.engine_file = "./superpoint/superpoint_v1.engine";
    config.input_tensor_names.push_back("input");
    config.output_tensor_names.push_back("scores");
    config.output_tensor_names.push_back("descriptors");

    std::shared_ptr<SuperPointC> superP = std::make_shared<SuperPointC>(&config);

    // superP.build(true);
    superP.get()->build();
    cv::Mat img = cv::Mat(256, 256, CV_8UC1, cv::Scalar::all(1));
    superP.get()->inference(img);

    return 0;
}