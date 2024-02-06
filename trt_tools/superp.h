#pragma once
#include "trtInterface.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

class SuperPointC : public trtInterface
{
private:
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

public:
    SuperPointC(Config *config);
    bool inference(cv::Mat &img);
    bool process_input(const BufferManager &buffers, const cv::Mat &image);

    bool process_output(const BufferManager &buffers);
};