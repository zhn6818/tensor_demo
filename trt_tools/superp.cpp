#include "superp.h"
#define DATA_SHAPE 256
SuperPointC::SuperPointC(Config *config) : trtInterface(config)
{
    trtInterface::initDims();
}

bool SuperPointC::inference(cv::Mat &img)
{
    if (!context_)
    {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(trtInterface::GetEngine()->createExecutionContext());
        if (!context_)
        {
            return false;
        }
    }
    assert(context_ != nullptr);
    assert(trtInterface::GetEngine()->getNbBindings() == 3);

    const int input_index = trtInterface::GetEngine()->getBindingIndex(trtInterface::config_->input_tensor_names[0].c_str());

    context_->setBindingDimensions(input_index, Dims4(1, 1, img.rows, img.cols));

    BufferManager buffers(trtInterface::GetEngine(), 0, context_.get());

    ASSERT(config_->input_tensor_names.size() == 1);
    if (!process_input(buffers, img))
    {
        return false;
    }
    buffers.copyInputToDevice();

    bool status = context_->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }
    buffers.copyOutputToHost();

    if (!process_output(buffers))
    {
        return false;
    }
    return true;
    // bool status = context_->executeV2(buffers.getDeviceBindings().data());
    // if (!status)
    // {
    //     return false;
    // }
    // buffers.copyOutputToHost();
    // if (!process_output(buffers, features))
    // {
    //     return false;
    // }
    // return true;

    std::cout << std::endl;
    return true;
}

bool SuperPointC::process_output(const BufferManager &buffers)
{
    auto *output_score = static_cast<float *>(buffers.getHostBuffer(config_->output_tensor_names[0]));
    auto *output_desc = static_cast<float *>(buffers.getHostBuffer(config_->output_tensor_names[1]));

    for (int i = 0; i < 100; i++)
    {
        std::cout << i << "  ___   " << (float)(output_score[i]) << "  " << (float)(output_desc[i]) << "  " << std::endl;
    }

    return true;
}

bool SuperPointC::process_input(const BufferManager &buffers, const cv::Mat &image)
{
    auto *host_data_buffer = static_cast<float *>(buffers.getHostBuffer(trtInterface::config_->input_tensor_names[0]));

    for (int row = 0; row < image.rows; ++row)
    {
        for (int col = 0; col < image.cols; ++col)
        {
            host_data_buffer[row * image.cols + col] = float(image.at<unsigned char>(row, col));
        }
    }
    return true;
}
// bool SuperPointC::build(bool isAddOptiProfile)
// {
//     return trtInterface::build(isAddOptiProfile);
// }