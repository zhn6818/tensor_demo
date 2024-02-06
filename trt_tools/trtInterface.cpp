#include "trtInterface.h"

trtInterface::trtInterface(Config *config) : config_(config), engine_(nullptr)
{
    setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}
void trtInterface::initDims()
{
    vecDims.push_back(Dims4(1, 1, 100, 100));
    vecDims.push_back(Dims4(1, 1, 500, 500));
    vecDims.push_back(Dims4(1, 1, 3000, 3000));
}

bool trtInterface::build(bool isAddOptiProfile)
{
    if (deserialize_engine())
    {
        std::cout << "file exists!!!!!!!" << std::endl;
        return true;
    }
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        std::cout << "builder create failed !!!!!!!!!!" << std::endl;
        return false;
    }
    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network) 
    {
        std::cout << "metwork create failed !!!!!!!!!!" << std::endl;
        return false;
    }
    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        std::cout << "config create failed !!!!!!!!!!" << std::endl;
        return false;
    }
    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
    {
        std::cout << "parser create failed !!!!!!!!!!" << std::endl;
        return false;
    }

    if (isAddOptiProfile)
    {
        auto profile = builder->createOptimizationProfile();
        if (!profile)
        {
            std::cout << "profile create failed !!!!!!!!!!" << std::endl;
            return false;
        }
        // std::cout << config_->input_tensor_names[0].c_str() << std::endl;
        profile->setDimensions(config_->input_tensor_names[0].c_str(),
                               OptProfileSelector::kMIN, vecDims[0]);
        profile->setDimensions(config_->input_tensor_names[0].c_str(),
                               OptProfileSelector::kOPT, vecDims[1]);
        profile->setDimensions(config_->input_tensor_names[0].c_str(),
                               OptProfileSelector::kMAX, vecDims[2]);
        config->addOptimizationProfile(profile);
    }

    auto constructed = construct_network(builder, network, config, parser);
    if (!constructed)
    {
        std::cout << "construct create failed !!!!!!!!!!" << std::endl;
        return false;
    }
    auto profile_stream = tensorrt_virgo_common::makeCudaStream();
    if (!profile_stream)
    {
        std::cout << "stream create failed !!!!!!!!!!" << std::endl;
        return false;
    }
    config->setProfileStream(*profile_stream); // 创建新的CUDA流
    TensorRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        std::cout << "plan create failed !!!!!!!!!!" << std::endl;
        return false;
    }
    TensorRTUniquePtr<IRuntime> runtime{createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime)
    {
        std::cout << "runtime create failed !!!!!!!!!!" << std::endl;
        return false;
    }
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_)
    {
        std::cout << "engine create failed !!!!!!!!!!" << std::endl;
        return false;
    }
    save_engine();
    return true;
}

bool trtInterface::construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder, TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network, TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config, TensorRTUniquePtr<nvonnxparser::IParser> &parser) const
{
    auto parsed = parser->parseFromFile(config_->onnx_file.c_str(),
                                        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        std::cout << "parsed create failed !!!!!!!!!!" << std::endl;
        return false;
    }
    config->setMaxWorkspaceSize(512_MiB);
    // config->setFlag(BuilderFlag::kFP16);
    config->setFlag(BuilderFlag::kTF32);
    // enableDLA(builder.get(), config.get(), super_point_config_.dla_core);
    return true;
}

void trtInterface::save_engine()
{
    if (config_->engine_file.empty())
    {
        return;
    }
    if (engine_ != nullptr)
    {
        nvinfer1::IHostMemory *data = engine_->serialize();
        std::ofstream file(config_->engine_file, std::ios::binary);
        if (!file)
        {
            return;
        }
        file.write(reinterpret_cast<const char *>(data->data()), data->size());
    }
}

bool trtInterface::deserialize_engine()
{
    std::ifstream file(config_->engine_file.c_str(), std::ios::binary);
    if (file.is_open())
    {
        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        char *model_stream = new char[size];
        file.read(model_stream, size);
        file.close();
        nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
        if (runtime == nullptr)
        {
            delete[] model_stream;
            return false;
        }
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
        if (engine_ == nullptr)
        {
            delete[] model_stream;
            return false;
        }
        delete[] model_stream;
        return true;
    }
    return false;
}

std::shared_ptr<nvinfer1::ICudaEngine> trtInterface::GetEngine()
{
    return this->engine_;
}