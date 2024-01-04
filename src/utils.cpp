#include <map>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <dirent.h>
#include <vector>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "acl/acl.h"
#include "utils.h"

using namespace std;

namespace {
const std::string g_imagePathSeparator = ",";
const int STAT_SUCCESS = 0;
const std::string g_fileSperator = "/";
const std::string g_pathSeparator = "/";
// output image prefix
const std::string g_outputFilePrefix = "out_";
}

bool Utils::IsDirectory(const string &path)
{
    // get path stat
    struct stat buf;
    if (stat(path.c_str(), &buf) != STAT_SUCCESS) {
        return false;
    }

    // check
    if (S_ISDIR(buf.st_mode)) {
        return true;
    } else {
    return false;
    }
}

bool Utils::IsPathExist(const string &path)
{
    ifstream file(path);
    if (!file) {
        return false;
    }
    return true;
}

void Utils::SplitPath(const string &path, vector<string> &path_vec)
{
    char *tmp_path = strtok(const_cast<char*>(path.c_str()), g_imagePathSeparator.c_str());
    while (tmp_path) {
        path_vec.emplace_back(tmp_path);
        tmp_path = strtok(nullptr, g_imagePathSeparator.c_str());
    }
}

void Utils::GetAllFiles(const string &path, vector<string> &file_vec)
{
    // split file path
    vector<string> path_vector;
    SplitPath(path, path_vector);

    for (string every_path : path_vector) {
        // check path exist or not
        if (!IsPathExist(path)) {
        ERROR_LOG("Failed to deal path=%s. Reason: not exist or can not access.",
                  every_path.c_str());
        continue;
        }
        // get files in path and sub-path
        GetPathFiles(every_path, file_vec);
    }
}

void Utils::GetPathFiles(const string &path, vector<string> &file_vec)
{
    struct dirent *dirent_ptr = nullptr;
    DIR *dir = nullptr;
    if (IsDirectory(path)) {
        dir = opendir(path.c_str());
        while ((dirent_ptr = readdir(dir)) != nullptr) {
            // skip . and ..
            if (dirent_ptr->d_name[0] == '.') {
            continue;
            }

            // file path
            string full_path = path + g_pathSeparator + dirent_ptr->d_name;
            // directory need recursion
            if (IsDirectory(full_path)) {
                GetPathFiles(full_path, file_vec);
            } else {
                // put file
                file_vec.emplace_back(full_path);
            }
        }
    } else {
        file_vec.emplace_back(path);
    }
}

void* Utils::CopyDataDeviceToLocal(void* deviceData, uint32_t dataSize)
{
    uint8_t* buffer = new uint8_t[dataSize];
    if (buffer == nullptr) {
        ERROR_LOG("New malloc memory failed");
        return nullptr;
    }

    aclError aclRet = aclrtMemcpy(buffer, dataSize, deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("Copy device data to local failed, aclRet is %d", aclRet);
        delete[](buffer);
        return nullptr;
    }

    return (void*)buffer;
}

void* Utils::CopyDataToDevice(void* data, uint32_t dataSize, aclrtMemcpyKind policy)
{
    void* buffer = nullptr;
    aclError aclRet = aclrtMalloc(&buffer, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("malloc device data buffer failed, aclRet is %d", aclRet);
        return nullptr;
    }

    aclRet = aclrtMemcpy(buffer, dataSize, data, dataSize, policy);
    if (aclRet != ACL_SUCCESS) {
        ERROR_LOG("Copy data to device failed, aclRet is %d", aclRet);
        (void)aclrtFree(buffer);
        return nullptr;
    }

    return buffer;
}

void* Utils::CopyDataDeviceToDevice(void* deviceData, uint32_t dataSize)
{
    return CopyDataToDevice(deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
}

void* Utils::CopyDataHostToDevice(void* deviceData, uint32_t dataSize)
{
    return CopyDataToDevice(deviceData, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
}

Result Utils::CopyImageDataToDevice(ImageData& imageDevice, ImageData srcImage, aclrtRunMode mode)
{
    void* buffer;
    if (mode == ACL_HOST)
        buffer = Utils::CopyDataHostToDevice(srcImage.data.get(), srcImage.size);
    else
        buffer = Utils::CopyDataDeviceToDevice(srcImage.data.get(), srcImage.size);

    if (buffer == nullptr) {
        ERROR_LOG("Copy image to device failed");
        return FAILED;
    }

    imageDevice.width = srcImage.width;
    imageDevice.height = srcImage.height;
    imageDevice.size = srcImage.size;
    imageDevice.data.reset((uint8_t*)buffer, [](uint8_t* p) { aclrtFree((void *)p); });

    return SUCCESS;
}

std::string Utils::ConvertNum2Str(double num)
{
    std::ostringstream oss;
    oss<<setiosflags(ios::fixed)<<std::setprecision(3)<<num;
    std::string str(oss.str());
    return str;
}

std::string Utils::ConvertDegreesNum2Str(double num, const char type)
{
    std::ostringstream oss;
    if (type == 'd') {
        oss << num;
    } else if (type == 'm') {
        if (abs(num) < 10) {
            oss <<std::setw(2)<<std::setfill('0')<<num;
        } else {
            oss << num;
        }
    } else {
        if (num == 0) {
            oss <<std::setw(1)<<std::setfill('0')<<num;
            oss<<setiosflags(ios::fixed)<<std::setprecision(2)<<num;
        } else if (ceil(num) == floor(num) && num != 0 && num < 10) {
            char strNum[64];
            sprintf(strNum, "%d%.2f\n", (int)num / 10, num);
            oss<<strNum;
        } else {
            oss<<setiosflags(ios::fixed)<<std::setprecision(2)<<num;
        }

    }

    std::string str(oss.str());
    return str;
}

void Utils::ConvertAngleUnits(const double input, int *degrees, int *minutes, int *seconds)
{
    *degrees = floor(input);
    *minutes = floor((input - *degrees) * 60);
    *seconds = (input - *degrees) * 3600 - *minutes * 60;
}

// 绘制度分秒 1°00′00.00″
std::string Utils::GetDegMinSec(const double inputAngle)
{
    int degrees = 0;
    int minutes = 0;
    int seconds = 0;
    ConvertAngleUnits(abs(inputAngle), &degrees, &minutes, &seconds);
    std::string currDeg = ConvertDegreesNum2Str(degrees, 'd') + "deg" + ConvertDegreesNum2Str(minutes, 'm') + "'" + ConvertDegreesNum2Str(seconds, 's') + "''";
    return currDeg;
}

std::string Utils::ConvertTimesNum2Str(double num)
{
    std::ostringstream oss;
    oss<<setiosflags(ios::fixed)<<std::setprecision(1)<<num;
    std::string str(oss.str());
    return str;
}
