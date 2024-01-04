#ifndef UTILS_H
#define UTILS_H

#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <unistd.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include "AclLiteType.h"
#include "acl/acl.h"

using namespace std;

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]   " fmt "\n", ##args)

#define RGBU8_IMAGE_SIZE(width, height) ((width) * (height) * 3)
#define YUV420SP_SIZE(width, height) ((width) * (height) * 3 / 2)

#define ALIGN_UP(num, align) (((num) + (align) - 1) & ~((align) - 1))
#define ALIGN_UP2(num) ALIGN_UP(num, 2)
#define ALIGN_UP16(num) ALIGN_UP(num, 16)
#define ALIGN_UP128(num) ALIGN_UP(num, 128)

#define SHARED_PTR_DVPP_BUF(buf) (shared_ptr<uint8_t>((uint8_t *)(buf), [](uint8_t* p) { acldvppFree(p); }))
#define SHARED_PTR_U8_BUF(buf) (shared_ptr<uint8_t>((uint8_t *)(buf), [](uint8_t* p) { delete[](p); }))

template<class Type>
std::shared_ptr<Type> MakeSharedNoThrow()
{
    try {
        return std::make_shared<Type>();
    }
    catch (...) {
        return nullptr;
    }
}

#define MAKE_SHARED_NO_THROW(memory, memory_type) \
    do { \
            memory = MakeSharedNoThrow<(memory_type)>(); \
    }while (0)

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
}Result;

/**
 * Utils
 */
class Utils {
public:

    /**
    * @brief create device buffer of pic
    * @param [in] picDesc: pic desc
    * @param [in] PicBufferSize: aligned pic size
    * @return device buffer of pic
    */
    static bool IsDirectory(const std::string &path);
    static bool IsPathExist(const std::string &path);
    static void SplitPath(const std::string &path, std::vector<std::string> &path_vec);
    static void GetAllFiles(const std::string &path, std::vector<std::string> &file_vec);
    static void GetPathFiles(const std::string &path, std::vector<std::string> &file_vec);
    static void* CopyDataToDevice(void* data, uint32_t dataSize, aclrtMemcpyKind policy);
    static void* CopyDataDeviceToLocal(void* deviceData, uint32_t dataSize);
    static void* CopyDataHostToDevice(void* deviceData, uint32_t dataSize);
    static void* CopyDataDeviceToDevice(void* deviceData, uint32_t dataSize);
    static int ReadImageFile(ImageData& image, std::string fileName);
    static Result CopyImageDataToDevice(ImageData& imageDevice, ImageData srcImage, aclrtRunMode mode);
    static std::string ConvertNum2Str(double num);
    static std::string ConvertDegreesNum2Str(double num, const char type);
    static void ConvertAngleUnits(const double input, int *degrees, int *minutes, int *seconds);
    // 绘制度分秒 1°00′00.00″
    static std::string GetDegMinSec(const double inputAngle);
    static std::string ConvertTimesNum2Str(double num);
};

#endif