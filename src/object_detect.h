#ifndef OBJECT_DETECT_H
#define OBJECT_DETECT_H

#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/types_c.h"
#include "utils.h"
#include "acl/acl.h"
#include "model_process.h"

using namespace std;

/**
* ObjectDetect
*/
class ObjectDetect {
public:
    ObjectDetect(const char* modelPath, uint32_t modelWidth,
    uint32_t modelHeight);
    ~ObjectDetect();
    // Inference initialization
    Result Init();
    // nference frame image preprocessing
    Result Preprocess(cv::Mat& frame);
    // Inference frame picture
    Result Inference(aclmdlDataset*& inferenceOutput);
    // Inference output post-processing
    Result Postprocess(cv::Mat& frame, aclmdlDataset* modelOutput);

private:
    // Initializes the ACL resource
    Result InitResource();
    // Loading reasoning model
    Result InitModel(const char* omModelPath);
    Result CreateModelInputdDataset();

    // Get data from model inference output aclmdlDataset to local
    void* GetInferenceOutputItem(uint32_t& itemDataSize,
    aclmdlDataset* inferenceOutput,
    uint32_t idx);

    // Release the requested resources
    void DestroyResource();

private:
    int32_t g_deviceId_;  // Device ID, default is 0
    ModelProcess g_model_; // Inference model instance

    const char* g_modelPath_; // Offline model file path
    uint32_t g_modelWidth_;   // The input width required by the model
    uint32_t g_modelHeight_;  // The model requires high input
    uint32_t g_imageDataSize_; // Model input data size
    void*    g_imageDataBuf_;      // Model input data cache
    uint32_t g_imageInfoSize_;
    void*    g_imageInfoBuf_;
    aclrtRunMode g_runMode_;   // Run mode, which is whether the current application is running on atlas200DK or AI1
    bool g_isInited_;     // Initializes the tag to prevent inference instances from being initialized multiple times
};

#endif