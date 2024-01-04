

#ifndef MODEL_PROCESS_H
#define MODEL_PROCESS_H

#pragma once
#include <iostream>
#include "utils.h"
#include "acl/acl.h"

/**
* ModelProcess
*/
class ModelProcess {
public:
    /**
    * @brief Constructor
    */
    ModelProcess();

    /**
    * @brief Destructor
    */
    ~ModelProcess();

    /**
    * @brief load model from file with mem
    * @param [in] modelPath: model path
    * @return result
    */
    Result LoadModelFromFileWithMem(const char *modelPath);

    /**
    * @brief release all acl resource
    */
    void DestroyResource();

    /**
    * @brief unload model
    */
    void Unload();

    /**
    * @brief create model desc
    * @return result
    */
    Result CreateDesc();

    /**
    * @brief destroy desc
    */
    void DestroyDesc();

    /**
    * @brief create model input
    * @param [in] inputDataBuffer: input buffer
    * @param [in] bufferSize: input buffer size
    * @return result
    */
    Result CreateInput(void *input1, size_t input1Size,
                       void* input2, size_t input2Size);

    /**
    * @brief destroy input resource
    */
    void DestroyInput();

    /**
    * @brief create output buffer
    * @return result
    */
    Result CreateOutput();

    /**
    * @brief destroy output resource
    */
    void DestroyOutput();

    /**
    * @brief model execute
    * @return result
    */
    Result Execute();

    /**
    * @brief get model output data
    * @return output dataset
    */
    aclmdlDataset *GetModelOutputData();

private:
    bool g_loadFlag_;  // model load flag
    uint32_t g_modelId_;
    void *g_modelMemPtr_;
    size_t g_modelMemSize_;
    void *g_modelWeightPtr_;
    size_t g_modelWeightSize_;
    aclmdlDesc *g_modelDesc_;
    aclmdlDataset *g_input_;
    aclmdlDataset *g_output_;
    bool g_isReleased_;
};

#endif