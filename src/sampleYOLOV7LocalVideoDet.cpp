#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"
#include "label.h"
#include "AclLiteVideoProc.h"
#include <opencv2/imgproc/types_c.h>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <yaml-cpp/yaml.h>

using namespace std;
using namespace cv;
typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

cv::Mat img; 
cv::Mat srcImage;

typedef struct BoundBox {
    float x;
    float y;
    float width;
    float height;
    float score;
    size_t classIndex;
    size_t index;
} BoundBox;

bool sortScore(BoundBox box1, BoundBox box2)
{
    return box1.score > box2.score;
}

class SampleYOLOV7 {
    public:
    SampleYOLOV7(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight);
    Result InitResource();
    Result ProcessInput(string testImgPath);
    Result Inference(std::vector<InferenceOutput>& inferOutputs);
    Result GetResult(std::vector<InferenceOutput>& inferOutputs, string imagePath, size_t imageIndex, bool release);
    ~SampleYOLOV7();
    private:
    void ReleaseResource();
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
    const char *modelPath_;
    int32_t modelWidth_;
    int32_t modelHeight_;
};

SampleYOLOV7::SampleYOLOV7(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight) :
                           modelPath_(modelPath), modelWidth_(modelWidth), modelHeight_(modelHeight)
{
}

SampleYOLOV7::~SampleYOLOV7()
{
    ReleaseResource();
}

Result SampleYOLOV7::InitResource()
{
    // init acl resource
    AclLiteError ret = aclResource_.Init();
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("resource init failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("get runMode failed, errorCode is %d", ret);
        return FAILED;
    }

    // init dvpp resource
    ret = imageProcess_.Init();
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("imageProcess init failed, errorCode is %d", ret);
        return FAILED;
    }

    // load model from file
    ret = model_.Init(modelPath_);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleYOLOV7::ProcessInput(string testImgPath)
{
    // read image from file
    ImageData image;
    AclLiteError ret = ReadJpeg(image, testImgPath);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("ReadJpeg failed, errorCode is %d", ret);
        return FAILED;
    }

    // copy image from host to dvpp
    ImageData imageDevice;
    ret = CopyImageToDevice(imageDevice, image, runMode_, MEMORY_DVPP);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("CopyImageToDevice failed, errorCode is %d", ret);
        return FAILED;
    }

    // image decoded from JPEG format to YUV
    ImageData yuvImage;
    ret = imageProcess_.JpegD(yuvImage, imageDevice);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("Convert jpeg to yuv failed, errorCode is %d", ret);
        return FAILED;
    }

    // zoom image to modelWidth_ * modelHeight_
    ret = imageProcess_.Resize(resizedImage_, yuvImage, modelWidth_, modelHeight_);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("Resize image failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

int sockfdUDPSend;
struct sockaddr_in serverAddrUdpSend;
void InitUDPSendInfo()
{
    YAML::Node config = YAML::LoadFile("config.yaml");
    const std::string serverIp = config["serverIp"].as<std::string>();
    
    // 定义服务器地址和端口
    uint16_t serverPort = config["serverPort"].as<uint16_t>();
    printf("serverIp:%s serverPort:%d\n", serverIp.c_str(), serverPort);

    // 创建UDP套接字
    sockfdUDPSend = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfdUDPSend < 0) {
        ACLLITE_LOG_INFO("Failed to create socket.");
        return;
    }

    // 设置服务器地址
    serverAddrUdpSend.sin_family = AF_INET;
    serverAddrUdpSend.sin_port = htons(serverPort);
    serverAddrUdpSend.sin_addr.s_addr = inet_addr(serverIp.c_str());
    bind(sockfdUDPSend, (sockaddr*)&serverAddrUdpSend, sizeof(serverAddrUdpSend));//绑定端口号
    ACLLITE_LOG_INFO("InitUDPSendInfo success");
}

const int UP_UDP_PACK_SIZE = 60000;
void UdpSendVideo(cv::Mat &image)
{
    // 将图像数据转换为字符数组
    std::vector<uchar> imageData;
    std::vector<int> quality;
    quality.push_back(cv::IMWRITE_JPEG_QUALITY);
    quality.push_back(30);//进行50%的压缩
    cv::imencode(".jpg", image, imageData, quality);//将图像编码

    int nSize = imageData.size();
    int total_pack = 1 + (nSize - 1) / UP_UDP_PACK_SIZE;
    int ibuf[1];
    ibuf[0] = total_pack;
    sendto(sockfdUDPSend,ibuf,sizeof(int),0,(sockaddr *)&serverAddrUdpSend,sizeof(serverAddrUdpSend));
    std::cout<<"pack number : "<<total_pack<<std::endl;
    for(int i = 0; i<total_pack; i++){
      sendto(sockfdUDPSend, &imageData[i * UP_UDP_PACK_SIZE], UP_UDP_PACK_SIZE, 0, (sockaddr *)&serverAddrUdpSend,sizeof(serverAddrUdpSend));
    }
    usleep(10000);
    ACLLITE_LOG_INFO("send image data\n.");
}

Result SampleYOLOV7::Inference(std::vector<InferenceOutput>& inferOutputs)
{
    // create input data set of model
    AclLiteError ret = model_.CreateInput(static_cast<void *>(img.data), 640*640*1.5);
    // AclLiteError ret = model_.CreateInput(static_cast<void *>(resizedImage_.data.get()), resizedImage_.size);
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }

    // inference
    ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }

    return SUCCESS;
}

Result SampleYOLOV7::GetResult(std::vector<InferenceOutput>& inferOutputs,
                               string imagePath, size_t imageIndex, bool release)
{
    uint32_t outputDataBufId = 0;
    float *classBuff = static_cast<float *>(inferOutputs[outputDataBufId].data.get());
    // confidence threshold
    float confidenceThreshold = 0.25;

    // class number
    size_t classNum = 80;

    // number of (x, y, width, hight, confidence)
    size_t offset = 5;

    // total number = class number + (x, y, width, hight, confidence)
    size_t totalNumber = classNum + offset;

    // total number of boxs
    size_t modelOutputBoxNum = 25200;

    // top 5 indexes correspond (x, y, width, hight, confidence),
    // and 5~85 indexes correspond object's confidence
    size_t startIndex = 5;

    // read source image from file
    // cv::Mat srcImage = cv::imread(imagePath);
    int srcWidth = srcImage.cols;
    int srcHeight = srcImage.rows;

    // filter boxes by confidence threshold
    vector <BoundBox> boxes;
    size_t yIndex = 1;
    size_t widthIndex = 2;
    size_t heightIndex = 3;
    size_t classConfidenceIndex = 4;
    for (size_t i = 0; i < modelOutputBoxNum; ++i) {
        float maxValue = 0;
        float maxIndex = 0;
        for (size_t j = startIndex; j < totalNumber; ++j) {
            float value = classBuff[i * totalNumber + j] * classBuff[i * totalNumber + classConfidenceIndex];
                if (value > maxValue) {
                // index of class
                maxIndex = j - startIndex;
                maxValue = value;
            }
        }
        float classConfidence = classBuff[i * totalNumber + classConfidenceIndex];
        if (classConfidence >= confidenceThreshold) {
            // index of object's confidence
            size_t index = i * totalNumber + maxIndex + startIndex;

            // finalConfidence = class confidence * object's confidence
            float finalConfidence =  classConfidence * classBuff[index];
            BoundBox box;
            box.x = classBuff[i * totalNumber] * srcWidth / modelWidth_;
            box.y = classBuff[i * totalNumber + yIndex] * srcHeight / modelHeight_;
            box.width = classBuff[i * totalNumber + widthIndex] * srcWidth/modelWidth_;
            box.height = classBuff[i * totalNumber + heightIndex] * srcHeight / modelHeight_;
            box.score = finalConfidence;
            box.classIndex = maxIndex;
            box.index = i;
            if (maxIndex < classNum) {
                boxes.push_back(box);
            }
        }
           }

    // filter boxes by NMS
    vector <BoundBox> result;
    result.clear();
    float NMSThreshold = 0.45;
    int32_t maxLength = modelWidth_ > modelHeight_ ? modelWidth_ : modelHeight_;
    std::sort(boxes.begin(), boxes.end(), sortScore);
    BoundBox boxMax;
    BoundBox boxCompare;
    while (boxes.size() != 0) {
        size_t index = 1;
        result.push_back(boxes[0]);
        while (boxes.size() > index) {
            boxMax.score = boxes[0].score;
            boxMax.classIndex = boxes[0].classIndex;
            boxMax.index = boxes[0].index;

            // translate point by maxLength * boxes[0].classIndex to
            // avoid bumping into two boxes of different classes
            boxMax.x = boxes[0].x + maxLength * boxes[0].classIndex;
            boxMax.y = boxes[0].y + maxLength * boxes[0].classIndex;
            boxMax.width = boxes[0].width;
            boxMax.height = boxes[0].height;

            boxCompare.score = boxes[index].score;
            boxCompare.classIndex = boxes[index].classIndex;
            boxCompare.index = boxes[index].index;

            // translate point by maxLength * boxes[0].classIndex to
            // avoid bumping into two boxes of different classes
            boxCompare.x = boxes[index].x + boxes[index].classIndex * maxLength;
            boxCompare.y = boxes[index].y + boxes[index].classIndex * maxLength;
            boxCompare.width = boxes[index].width;
            boxCompare.height = boxes[index].height;

            // the overlapping part of the two boxes
            float xLeft = max(boxMax.x, boxCompare.x);
            float yTop = max(boxMax.y, boxCompare.y);
            float xRight = min(boxMax.x + boxMax.width, boxCompare.x + boxCompare.width);
            float yBottom = min(boxMax.y + boxMax.height, boxCompare.y + boxCompare.height);
            float width = max(0.0f, xRight - xLeft);
            float hight = max(0.0f, yBottom - yTop);
            float area = width * hight;
            float iou =  area / (boxMax.width * boxMax.height + boxCompare.width * boxCompare.height - area);

            // filter boxes by NMS threshold
            if (iou > NMSThreshold) {
                boxes.erase(boxes.begin() + index);
                continue;
            }
            ++index;
        }
        boxes.erase(boxes.begin());
    }

    // opencv draw label params
    const double fountScale = 0.5;
    const uint32_t lineSolid = 2;
    const uint32_t labelOffset = 11;
    const cv::Scalar fountColor(0, 0, 255);
    const vector <cv::Scalar> colors{
        cv::Scalar(237, 149, 100), cv::Scalar(0, 215, 255),
        cv::Scalar(50, 205, 50), cv::Scalar(139, 85, 26)};

    int half = 2;
    for (size_t i = 0; i < result.size(); ++i) {
        cv::Point leftUpPoint, rightBottomPoint;
        leftUpPoint.x = result[i].x - result[i].width / half;
        leftUpPoint.y = result[i].y - result[i].height / half;
        rightBottomPoint.x = result[i].x + result[i].width / half;
        rightBottomPoint.y = result[i].y + result[i].height / half;
        cv::rectangle(srcImage, leftUpPoint, rightBottomPoint, colors[i % colors.size()], lineSolid);
        string className = label[result[i].classIndex];
        string markString = to_string(result[i].score) + ":" + className;
        cv::putText(srcImage, markString, cv::Point(leftUpPoint.x, leftUpPoint.y + labelOffset),
                    cv::FONT_HERSHEY_COMPLEX, fountScale, fountColor);
    }
    // string savePath = "out_" + to_string(imageIndex) + ".jpg";
    // cv::imwrite(savePath, srcImage);
    UdpSendVideo(srcImage);
    // cv::imshow("1", srcImage);
    // cv::waitKey(1);
    if (release){
        free(classBuff);
        classBuff = nullptr;
    }
    return SUCCESS;
}

void SampleYOLOV7::ReleaseResource()
{
    model_.DestroyResource();
    imageProcess_.DestroyResource();
    aclResource_.Release();
}

bool IsHeader(uint8_t* buf)
{
    for(int i=0;i<4;i+=4)
    {
        // if(!(buf[i] == 0xdf && buf[i+1] == 0x0d && buf[i+2] == 0x76 && buf[i+3] == 0x7b))
        if(!(buf[i] == 0x7b && buf[i+1] == 0x76 && buf[i+2] == 0x0d && buf[i+3] == 0xdf))
        {
            return false;
        }
    }
    // int i = 0xdf0d767b;
    // if(i == *((int*)buf))
    //     return true;
    // else
    //     return false;

    return true;
}

int IMG_FRAME_SIZE = 640*512;

int main(int argc, char *argv[])
{
    // int sockfd;
    // struct sockaddr_in serverAddr, clientAddr;
    // uint8_t buffer[1024];
    // uint8_t* imgBufRaw;

    // imgBufRaw = (uint8_t*)malloc(IMG_FRAME_SIZE*sizeof(uint8_t));

    // int sum = 0;
    // ssize_t dataSize = 0;

    // // 创建UDP套接字
    // sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    // if (sockfd < 0) {
    //     std::cerr << "Failed to create socket" << std::endl;
    //     return 1;
    // }

    // // 设置服务器地址信息
    // serverAddr.sin_family = AF_INET;
    // serverAddr.sin_port = htons(10001); // 指定UDP端口
    // serverAddr.sin_addr.s_addr = INADDR_ANY;

    // // 将套接字绑定到服务器地址
    // if (bind(sockfd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
    //     std::cerr << "Failed to bind socket" << std::endl;
    //     return 1;
    // }

    // std::cout << "Waiting for data..." << std::endl;

    // bool findHeader = false;
    // int frameCnt = 0;

    // auto stamp = std::chrono::system_clock::now();
    // auto laststamp = std::chrono::system_clock::now();

    // int ii = 0;
    // bool started = false;

    // int nRecvBuf = 5 * 1024 * 1024;       
    // setsockopt(sockfd,SOL_SOCKET, SO_RCVBUF, (const char *)&nRecvBuf,sizeof(nRecvBuf));

    // cv::Mat dispimg;
    // // cv::Mat img = cv::Mat(512,640,CV_16UC1);
    // cv::Mat udpimg = cv::Mat(512,640,CV_8UC1);

    const char* modelPath = "/det.om";
    const string imagePath = "../data";
    const int32_t modelWidth = 640;
    const int32_t modelHeight = 640;

    // all images in dir
    // DIR *dir = opendir(imagePath.c_str());
    // if (dir == nullptr)
    // {
    //     ACLLITE_LOG_ERROR("file folder does no exist, please create folder %s", imagePath.c_str());
    //     return FAILED;
    // }
    vector<string> allPath;
    // struct dirent *entry;
    // while ((entry = readdir(dir)) != nullptr)
    // {
    //     if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0
    //     || strcmp(entry->d_name, ".keep") == 0)
    //     {
    //         continue;
    //     }else{
    //         string name = entry->d_name;
    //         string imgDir = imagePath +"/"+ name;
    //         allPath.push_back(imgDir);
    //     }
    // }
    // closedir(dir);

    // if (allPath.size() == 0){
    //     ACLLITE_LOG_ERROR("the directory is empty, please download image to %s", imagePath.c_str());
    //     return FAILED;
    // }

    // inference
    string fileName;
    bool release = false;
    SampleYOLOV7 sampleYOLO(modelPath, modelWidth, modelHeight);
    Result ret = sampleYOLO.InitResource();
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
        return FAILED;
    }

    // AclLiteVideoProc* cap = new AclLiteVideoProc("/root/samples/inference/modelInference/sampleYOLOV7/4_1.mp4");
    // if(!cap->IsOpened()) {
    //     ACLLITE_LOG_ERROR("Open camera failed");
    //     return ACLLITE_ERROR;
    // }
    // while(true) {
    //     ImageData image;
    //     int ret = cap->Read(image);
    //     printf("read\n");
    //     if (ret) {
    //         ACLLITE_LOG_ERROR("Read image failed, error %d", ret);
    //         break;
    //     }
    // }

    std::string videopath = "/home/HwHiAiUser/Videos/1.mp4";
    if(argc > 1)
    {
        videopath = argv[1];
        ACLLITE_LOG_INFO("video:%s\n", argv[1]);
    }

    cv::VideoCapture *cvcap = new cv::VideoCapture(videopath);
    std::vector<InferenceOutput> inferOutputs;

    InitUDPSendInfo();

    /*  local video deal*/
    while(1)
    {
        *cvcap>>img;
        if(img.empty())
            continue;
        // UdpSendVideo(img);
        // // cv::imshow("1", img);
        // // cv::waitKey(1);
        // continue;
        srcImage = img.clone();
        if (img.rows <= 0) {
            continue;
        }

        cv::resize(img, img,cv::Size(640,640));
        cv::cvtColor(img,img,CV_BGR2YUV_I420);
        
        // cv::imwrite("1.jpg", img);
        // return 0;
        inferOutputs.clear();
        ret = sampleYOLO.Inference(inferOutputs);
        if (ret == FAILED) {
            ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
            return FAILED;
        }

        int i=0;
        ret = sampleYOLO.GetResult(inferOutputs, fileName, i, release);
        if (ret == FAILED) {
            ACLLITE_LOG_ERROR("GetResult failed, errorCode is %d", ret);
            return FAILED;
        }
    }

    return 0;
    
    // int i=0;
    // while(1)
    // {
    //     sum = 0;
    //     laststamp = std::chrono::system_clock::now();
    //     while (true) {
    //         socklen_t clientAddrLen = sizeof(clientAddr);

    //         // printf("recvfrom:\n");
    //         dataSize = recvfrom(sockfd, buffer, sizeof(buffer), 0, (struct sockaddr *)&clientAddr, &clientAddrLen);
    //         if (dataSize < 0) {
    //             std::cerr << "Failed to receive data" << std::endl;
    //             return 1;
    //         }
    //         if(!findHeader)
    //         {
    //             // printf("******in find header*******\n");
    //             if(IsHeader(buffer))
    //             {
    //                 printf("******find header*******\n");
    //                 // printf("sum:%d\n", sum);
    //                 findHeader = true;
    //                 continue;
    //             }
    //         }
    //         else
    //         {
    //             memcpy(imgBufRaw+sum, buffer, dataSize);
    //             sum += dataSize;

    //             if(sum == IMG_FRAME_SIZE)
    //             // if(sum == 656384)
    //             // if(IsHeader(buffer))
    //             {
    //                 printf("find second header, sum:%d\n", sum);
    //                 findHeader = false;
    //                 break;
    //             }

    //         }
    //     }

    //     // printf("header:%#x,%#x,%#x,%#x\n", imgBufRaw[0],imgBufRaw[1],imgBufRaw[2],imgBufRaw[3]);
    //     if(sum == IMG_FRAME_SIZE)
    //     {
    //         // printf("frame cnt:%d\n", frameCnt++);
            
    //         udpimg.data = imgBufRaw;

    //         stamp = std::chrono::system_clock::now();

    //         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stamp - laststamp);

    //         laststamp = stamp;

    //         std::cout<<"time interval:"<< duration.count() << "ms"<<std::endl;

    //         // cv::imwrite("ret.png", img);
    //         UdpSendVideo(udpimg);
    //         // cv::imshow("1", udpimg);

    //         // srcImage = udpimg.clone();
    //         // cv::resize(udpimg,img,cv::Size(640,640));
    //         // cv::cvtColor(img,img,CV_BGR2YUV_I420);
            
    //         // // cv::imwrite("1.jpg", img);
    //         // // return 0;

    //         // ret = sampleYOLO.Inference(inferOutputs);
    //         // if (ret == FAILED) {
    //         //     ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
    //         //     return FAILED;
    //         // }

    //         // ret = sampleYOLO.GetResult(inferOutputs, fileName, i, release);
    //         // if (ret == FAILED) {
    //         //     ACLLITE_LOG_ERROR("GetResult failed, errorCode is %d", ret);
    //         //     return FAILED;
    //         // }

    //         //  writer << dispimg;
    //         // if(ii%15 == 0)
    //         // {
    //         //     cv::imwrite(std::to_string(ii/15)+".png", dispimg);
    //         // }
    //         // ii++;
    //         // printf("ii:%d\n", ii);
            
    //         // cv::waitKey(1);
    //         memset(imgBufRaw, 0, IMG_FRAME_SIZE*sizeof(uint8_t));
             

    //     } else {
    //         printf("error img data, sum:%d\n", sum);
    //     }

    //     // if (allPath.size() == i){
    //     //     release = true;
    //     // }
        
    //     // fileName = allPath.at(i).c_str();

    //     // ret = sampleYOLO.ProcessInput(fileName);
    //     // if (ret == FAILED) {
    //     //     ACLLITE_LOG_ERROR("ProcessInput image failed, errorCode is %d", ret);
    //     //     return FAILED;
    //     // }

        
    //     // return 0;
    // }
    close(sockfdUDPSend);
    return SUCCESS;
}
