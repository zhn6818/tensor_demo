#pragma once
#include <iostream>
#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <fstream>
#include <memory>
#include "trt_logger.h"
#include <iostream>
#include <Eigen/Core>
#include "NvInferRuntime.h"
#include "trt_config.h"
#include "trt_common.h"
#include "buffers.h"

using namespace tensorrt_virgo_log;
using namespace tensorrt_virgo_buffer;
using tensorrt_virgo_common::TensorRTUniquePtr;

class trtInterface
{
private:
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    
    // nvinfer1::Dims input_dims_{};
    std::vector<Dims4> vecDims;
    

public:
    Config *config_;
    trtInterface(Config *super_point_config);
    virtual void initDims();
    virtual bool build(bool isAddOptiProfile = true);
    void save_engine();
    bool deserialize_engine();
    bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder, TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network, TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config, TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;
    std::shared_ptr<nvinfer1::ICudaEngine> GetEngine();
};