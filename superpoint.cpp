#include <iostream>
#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <fstream>
#include <iostream>

#define BATCH_SIZE 1
#define DATA_SHAPE 256

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING) : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char *msg) noexcept
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};

void onnxTotrt(const std::string &model_file,            // name of the onnx model
               nvinfer1::IHostMemory **trt_model_stream, // output buffer for the TensorRT model
               Logger g_logger_)
{

    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);

    // -- create the builder ------------------/
    const auto explicit_batch = static_cast<uint32_t>(BATCH_SIZE)
                                << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(g_logger_);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicit_batch);

    // --create the parser to load onnx file---/
    auto parser = nvonnxparser::createParser(*network, g_logger_);
    if (!parser->parseFromFile(model_file.c_str(), verbosity))
    {
        std::string msg("failed to parse onnx file");
        g_logger_.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    // -- build the config for pass in specific parameters ---/
    builder->setMaxBatchSize(BATCH_SIZE);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();

    auto profile = builder->createOptimizationProfile();
    if (!profile)
    {
        std::cout << "profile error" << std::endl;
    }
    profile->setDimensions("input",
                           nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 1, 100, 100));
    profile->setDimensions("input",
                           nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 1, 500, 500));
    profile->setDimensions("input",
                           nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 1, 3000, 3000));
    config->addOptimizationProfile(profile);
    // config->setMaxWorkspaceSize(1 << 20);
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);

    // std::cout <<"engine bindings dimension" << engine->getNbBindings() << std::endl;

    // -- serialize the engine,then close everything down --/
    *trt_model_stream = engine->serialize();

    parser->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
};

int main(int argc, char **argv)
{

    Logger g_logger_;
    nvinfer1::IHostMemory *trt_model_stream{nullptr};
    std::string onnx_file = "./superpoint/superpoint_v1.onnx";

    // --Pass the params recorded in ONNX_file to trt_model_stream --/
    onnxTotrt(onnx_file, &trt_model_stream, g_logger_);

    std::ofstream p("./superpoint/superpoint_v1.engine", std::ios::binary);
    if (!p)
    {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char *>(trt_model_stream->data()), trt_model_stream->size());

    nvinfer1::IRuntime *engine_runtime = nvinfer1::createInferRuntime(g_logger_);
    if (engine_runtime == nullptr)
    {
        std::cerr << "Failed to create TensorRT Runtime object." << std::endl;
    }

    nvinfer1::ICudaEngine *engine_infer = engine_runtime->deserializeCudaEngine(trt_model_stream->data(), trt_model_stream->size(), nullptr);
    if (engine_infer == nullptr)
    {
        std::cerr << "Failed to create TensorRT Engine." << std::endl;
    }

    nvinfer1::IExecutionContext *engine_context = engine_infer->createExecutionContext();
    if (engine_context == nullptr)
    {
        std::cerr << "Failed to create TensorRT Execution Context." << std::endl;
    }

    // --destroy stream ---/.
    trt_model_stream->destroy();
    std::cout << "loaded trt model , do inference" << std::endl;

    ///////////////////////////////////////////////////////////////////
    // enqueue them up
    //////////////////////////////////////////////////////////////////

    // -- allocate host memory ------------/
    float h_input[DATA_SHAPE * DATA_SHAPE * 3] = {0.f};
    for (int i = 0; i < DATA_SHAPE * DATA_SHAPE * 3; i++)
    {
        h_input[i] = 1.0;
    }

    const int input_index = engine_infer->getBindingIndex("input");
    std::cout << "input_index: " << input_index << std::endl;
    engine_context->setBindingDimensions(input_index, nvinfer1::Dims4(1, 1, DATA_SHAPE, DATA_SHAPE));
    void *buffers[3];
    cudaMalloc(&buffers[0], DATA_SHAPE * DATA_SHAPE * 3 * sizeof(float)); //<- input
    cudaMalloc(&buffers[1], DATA_SHAPE * DATA_SHAPE * sizeof(float));
    cudaMalloc(&buffers[2], (DATA_SHAPE / 8) * (DATA_SHAPE / 8) * 256 * sizeof(float));

    float h_output1[DATA_SHAPE * DATA_SHAPE];
    float h_output2[(DATA_SHAPE / 8) * (DATA_SHAPE / 8) * 256];

    cudaMemcpy(buffers[0], h_input, DATA_SHAPE * DATA_SHAPE * 3 * sizeof(float), cudaMemcpyHostToDevice);
    int32_t BATCH_SIZE_ = 1;
    engine_context->execute(BATCH_SIZE_, buffers);
    cudaMemcpy(h_output1, buffers[1],
               DATA_SHAPE * DATA_SHAPE * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output2, buffers[2],
               (DATA_SHAPE / 8) * (DATA_SHAPE / 8) * 256 * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < 100; i++)
    {
        std::cout << i << "  ___   " << (float)(h_output1[i]) << "  " << (float)(h_output2[i]) << "  " << std::endl;
    }
    std::cout << "\n";
    return 0;
}