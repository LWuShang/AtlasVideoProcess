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
#include <thread>
#include "serial.h"

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
    //ACLLITE_LOG_INFO("send image data\n");
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

/***********************串口通信处理函数部分****************************/
Serial *obj = nullptr;

uint64_t createtimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    uint64_t* ptr=reinterpret_cast<uint64_t*>(&duration);
    uint64_t  tmp=*ptr;
    return tmp;
}

// CRC校验
unsigned short MakeCRC_R_LookupShortTable(uint8_t *LpDate, uint8_t Len)
{
    // 函数MakeCRC_R_LookupShortTable使用了一个静态的无符号短整型数组CRC_R_shortTable作为CRC校验码的查找表。查找表存储了预先计算好的校验码值，用于加快校验码的计算过程。
    static unsigned short CRC_R_shortTable[16] = {
        0x0000, 0x1081, 0x2102, 0x3183,
        0x4204, 0x5285, 0x6306, 0x7387,
        0x8408, 0x9489, 0xa50a, 0xb58b,
        0xc60c, 0xd68d, 0xe70e, 0xf78f};

    uint8_t da;
    unsigned short crc_reg = 0xffff;                         // 初始化crc_reg为0xFFFF，作为初始的CRC寄存器值。

    while (Len--)                                             // 循环迭代直到Len为0，每次迭代处理一个字节的数据。
    {
        da = (uint8_t)(crc_reg & 0x000f);                // 按位与操作提取crc_reg的低4位，并将结果转换为unsigned char类型
        crc_reg >>= 4;                                        // 右移4位，将crc_reg的高4位移到低4位
        crc_reg ^= CRC_R_shortTable[da ^ (*LpDate & 0x0f)]; // 根据查找表CRC_R_shortTable，更新crc_reg的值
                                                            // 取 da 和 LpDate 所指向字节的低4位进行异或操作，并查找CRC_R_shortTable中对应位置的值进行异或操作
        da = (uint8_t)(crc_reg & 0x000f);                // 再次进行上述操作，将 da 和 LpDate 所指向字节的高4位进行异或和查找表操作
        crc_reg >>= 4;
        crc_reg ^= CRC_R_shortTable[da ^ (*LpDate / 16)];

        LpDate++;                                             // 移动指针LpDate，指向下一个字节。
    }

    return (crc_reg ^ 0xffff);                                 // 将crc_reg与0xFFFF做异或操作，返回计算得到的CRC校验码。
}


//帧数据处理
void processData(const uint8_t buf[]) {
    SelfCheckInstruction selfCheck;
    selfCheck.PowerOnSelfCheck = buf[7] & 0x01;
    selfCheck.PeriodicSelfCheck = (buf[7] & 0x02) >> 1;  //  (buf[7] & 0x02)?1:0
    selfCheck.ReadProductIdentifier = (buf[7] & 0x04) >> 2;  // (buf[7] & 0x04)?1:0
    if(selfCheck.PowerOnSelfCheck) cout << "上电自检" << endl;
    if(selfCheck.PeriodicSelfCheck) cout << "周期自检" << endl;
    if(selfCheck.ReadProductIdentifier) cout << "读取产品标识码" << endl;

    ModeControlInstruction modectrl = ModeControlInstruction(buf[8]);
    switch (modectrl) {
        case ModeControlInstruction::FollowSearch:
            cout << "随动搜索" << endl;
            break;
        case ModeControlInstruction::AutoSearch:
            cout << "自动搜索" << endl;
            break;
        case ModeControlInstruction::ManualSearch:
            cout << "手动搜索" << endl;
            break;
        case ModeControlInstruction::SnowPlowSearch:
            cout << "雪犁搜索" << endl;
            break;
        case ModeControlInstruction::VideoTracking:
            cout << "视频跟踪" << endl;
            break;
        case ModeControlInstruction::LightSpotSearch:
            cout << "光斑搜索" << endl;
            break;
        case ModeControlInstruction::LightSpotTracking:
            cout << "光斑跟踪" << endl;
            break;
        case ModeControlInstruction::GeographicTrackingMode:
            cout << "地理跟踪模式" << endl;
            break;
        case ModeControlInstruction::MapMatching:
            cout << "地图匹配" << endl;
            break;
        default:
            cout << "无效" << endl;
    }

    SendTemplateInstruction STI = SendTemplateInstruction(buf[9]);
    if (STI == SendTemplateInstruction::SendTemplate) {
        cout << "发送目标模板" << endl;
    }
    else cout << "无效" << endl;


    //视场图像状态指令
    uint8_t VisualFieldStatusDirective = buf[11];
    int bit0to1 = VisualFieldStatusDirective & 0x03;
    if(bit0to1 == 0) cout << "无效" << endl;
    if(bit0to1 == 1) cout << "大视场" << endl;
    if(bit0to1 == 2) cout << "小视场" << endl;

    int bit2to3 = (VisualFieldStatusDirective >> 2) & 0x03;
    if(bit2to3 == 0) cout << "无效" << endl;
    if(bit2to3 == 1) cout << "数字倍数+" << endl;
    if(bit2to3 == 2) cout << "数字倍数-" << endl;

    int bit4to5 = (VisualFieldStatusDirective >> 4) & 0x03;
    if(bit4to5 == 0) cout << "无效" << endl;
    if(bit4to5 == 1) cout << "去雾档位+" << endl;
    if(bit4to5 == 2) cout << "去雾档位-" << endl;

    int bit6to7 = (VisualFieldStatusDirective >> 6) & 0x03;
    if(bit6to7 == 0) cout << "无效" << endl;
    if(bit6to7 == 1) cout << "增强档位+ " << endl;
    if(bit6to7 == 2) cout << "增强档位-" << endl;

    VisualFieldStatusDirective = buf[12];
    bit0to1 = VisualFieldStatusDirective & 0x03;
    if(bit0to1 == 0) cout << "无效" << endl;
    if(bit0to1 == 1) cout << "降噪档位+" << endl;
    if(bit0to1 == 2) cout << "降噪档位-" << endl;

    bit2to3 = (VisualFieldStatusDirective >> 2) & 0x03;
    if(bit2to3 == 0) cout << "无效" << endl;
    if(bit2to3 == 1) cout << "白热" << endl;
    if(bit2to3 == 2) cout << "黑热" << endl;

    int bit4to6 = (VisualFieldStatusDirective >> 4) & 0x07;
    if(bit4to6 == 0) cout << "无效" << endl;
    if(bit4to6 == 1) cout << "x向+" << endl;
    if(bit4to6 == 2) cout << "x向-" << endl;
    if(bit4to6 == 3) cout << "y向+" << endl;
    if(bit4to6 == 4) cout << "y向-" << endl;

    //识别开关
    uint8_t RecognitionSwitch = buf[13];
    bit0to1 = RecognitionSwitch & 0x03;
    if(bit0to1 == 0) cout << "无效" << endl;
    if(bit0to1 == 1) cout << "透传无识别(隐藏识别信息的字符,并保持SRIO识别信息上传)" << endl;
    if(bit0to1 == 2) cout << "识别" << endl;

    int bit2to4 = (RecognitionSwitch >> 2) & 0x07;
    if(bit2to4 == 0) cout << "未知" << endl;
    if(bit2to4 == 1) cout << "停机坪飞机" << endl;
    if(bit2to4 == 2) cout << "地面FKLD车" << endl;
    if(bit2to4 == 3) cout << "地空DD车" << endl;
    if(bit2to4 == 4) cout << "ZJ车" << endl;
    if(bit2to4 == 5) cout << "LDZD" << endl;
    if(bit2to4 == 6) cout << "机场" << endl;


    //异常信息1
    uint8_t ExceptionMessage1 = buf[14];
    bit0to1 = ExceptionMessage1 & 0x03;
    if(bit0to1 == 0) cout << "无效" << endl;
    if(bit0to1 == 1) cout << "伺服转台正常" << endl;
    if(bit0to1 == 2) cout << "伺服转台异常" << endl;

    bit2to3 = (ExceptionMessage1 >> 2) & 0x03;
    if(bit2to3 == 0) cout << "无效" << endl;
    if(bit2to3 == 1) cout << "测距组件正常" << endl;
    if(bit2to3 == 2) cout << "测距组件异常" << endl;

    bit4to5 = (ExceptionMessage1 >> 4) & 0x03;
    if(bit4to5 == 0) cout << "无效" << endl;
    if(bit4to5 == 1) cout << "光斑跟踪组件正常" << endl;
    if(bit4to5 == 2) cout << "光斑跟踪组件异常" << endl;

    bit6to7 = (ExceptionMessage1 >> 6) & 0x03;
    if(bit6to7 == 0) cout << "无效" << endl;
    if(bit6to7 == 1) cout << "激光照射器正常" << endl;
    if(bit6to7 == 2) cout << "激光照射器异常" << endl;


    //异常信息2
    uint8_t ExceptionMessage2 = buf[15];
    bit0to1 = ExceptionMessage2 & 0x03;
    if(bit0to1 == 0) cout << "无效" << endl;
    if(bit0to1 == 1) cout << "位姿组件正常" << endl;
    if(bit0to1 == 2) cout << "位姿组件异常" << endl;

    bit2to3 = (ExceptionMessage2 >> 2) & 0x03;
    if(bit2to3 == 0) cout << "无效" << endl;
    if(bit2to3 == 1) cout << "共孔径接收子系统正常" << endl;
    if(bit2to3 == 2) cout << "共孔径接收子系统异常" << endl;

    bit4to5 = (ExceptionMessage2 >> 4) & 0x03;
    if(bit4to5 == 0) cout << "无效" << endl;
    if(bit4to5 == 1) cout << "白光电视较轴模块正常" << endl;
    if(bit4to5 == 2) cout << "白光电视较轴模块异常" << endl;


    //异常信息3
    uint8_t ExceptionMessage3 = buf[16];
    bit0to1 = ExceptionMessage3 & 0x03;
    if(bit0to1 == 0) cout << "无效" << endl;
    if(bit0to1 == 1) cout << "环控组件正常" << endl;
    if(bit0to1 == 2) cout << "环控组件异常" << endl;

    bit2to3 = (ExceptionMessage3 >> 2) & 0x03;
    if(bit2to3 == 0) cout << "无效" << endl;
    if(bit2to3 == 1) cout << "电源组件正常" << endl;
    if(bit2to3 == 2) cout << "电源组件异常" << endl;

    bit4to5 = (ExceptionMessage3 >> 4) & 0x03;
    if(bit4to5 == 0) cout << "无效" << endl;
    if(bit4to5 == 1) cout << "红外相机正常" << endl;
    if(bit4to5 == 2) cout << "红外相机异常" << endl;


    //提示指令
    uint8_t PromptInstruction = buf[17];
    bit0to1 = PromptInstruction & 0x03;
    if(bit0to1 == 0) cout << "无效" << endl;
    if(bit0to1 == 1) cout << "自动校轴" << endl;
    if(bit0to1 == 2) cout << "手动校轴" << endl;

    bit2to3 = (PromptInstruction >> 2) & 0x03;
    if(bit2to3 == 0) cout << "无效" << endl;
    if(bit2to3 == 1) cout << "自动对焦" << endl;
    if(bit2to3 == 2) cout << "手动对焦" << endl;

    int bit4 = (PromptInstruction & 0x10) ? 1 : 0;
    if(bit4) cout << "手动光电复位" << endl;
    else cout << "无效" << endl;

    int bit5 = (PromptInstruction & 0x20) ? 1 : 0;
    if(bit5) cout << "非均匀校正" << endl;
    else cout << "无效" << endl;

    int bit6 = (PromptInstruction & 0x40) ? 1 : 0;
    if(bit6) cout << "激光测距" << endl;
    else cout << "无效" << endl;

    int bit7 = (PromptInstruction & 0x80) ? 1 : 0;
    if(bit7) cout << "激光照射" << endl;
    else cout << "无效" << endl;

    PromptInstruction = buf[18];
    bit0to1 = PromptInstruction & 0x03;
    if(bit0to1 == 0) cout << "无效" << endl;
    if(bit0to1 == 1) cout << "主控接收目标模板成功" << endl;
    if(bit0to1 == 2) cout << "主控接收目标模板失败" << endl;

    bit2to3 = (PromptInstruction >> 2) & 0x03;
    if(bit2to3 == 0) cout << "无效" << endl;
    if(bit2to3 == 1) cout << "主控正在通过1553B向任务机发送模板" << endl;
    if(bit2to3 == 2) cout << "主控向任务机发送模板成功" << endl;
    if(bit2to3 == 3) cout << "主控向任务机发送模板失败" << endl;

    bit4to5 = (PromptInstruction >> 4) & 0x03;
    if(bit4to5 == 0) cout << "无效" << endl;
    if(bit4to5 == 1) cout << "延时跟踪补充失败" << endl;
    if(bit4to5 == 2) cout << "跟踪到目标" << endl;
    if(bit2to3 == 4) cout << "丢失目标" << endl;

    //帧时间
    string FrameTime;
    uint16_t year = (buf[19] << 8) | buf[20];

    //GNSS时间变换   年
    double j = floor(year + 0.5);
    double N = floor(4 * (j + 68549) / 146097);
    double L1 = j + 68569 - floor((N * 146097 + 3) / 4);
    double Y1 = floor(4000 * (L1 + 1) / 1461001);
    double L2 = L1 - floor((1461 * Y1) / 4) + 31;
    double M1 = floor(80 * L2 / 2447);
    double D = L2 - floor(2447 * M1 / 80);
    double L3 = floor(M1 / 11);
    int Y = floor(100 * (N - 49) + Y1 + L3);

    uint8_t month = buf[21];
    uint8_t day = buf[22];
    uint8_t hours = buf[23];
    uint8_t minutes = buf[24];
    uint8_t second = buf[25];
    uint8_t millisecond = buf[26];

    printf("%4d年%d月%d日\t%d:%d:%d:%d\n", Y, month, day,hours, minutes, second,millisecond);

    //激光测距时间
    uint8_t distanceMeasurementTime = buf[27];
    cout << distanceMeasurementTime << "s" << endl;

    //激光测距频率
    uint8_t distanceMeasurementFrequency = buf[28];
    if( distanceMeasurementFrequency == 0x01) cout << "1Hz" << endl;
    else if(distanceMeasurementFrequency == 0x05) cout << "5Hz" << endl;
    else if(distanceMeasurementFrequency == 0xFF) cout << "测距终止" << endl;
    else if(distanceMeasurementFrequency == 0x00) cout << "无效" << endl;

    //激光测距结果
    uint16_t distanceMeasurementResult = (buf[29] << 8) | buf[30];
    cout << "激光测距结果:" << distanceMeasurementResult << "m" << endl;

    //激光照射时间
    uint8_t laserIlluminationTime = buf[31];
    cout << "激光照射时间:" << laserIlluminationTime << "s" << endl;

    //激光照射频率
    uint8_t laserIlluminationFrequency = buf[32];
    if( laserIlluminationFrequency == 0x01) cout << "1Hz" << endl;
    else if(laserIlluminationFrequency == 0x05) cout << "5Hz" << endl;
    else if(laserIlluminationFrequency == 0xFF) cout << "照射终止" << endl;
    else if(laserIlluminationFrequency == 0x00) cout << "无效" << endl;

    //十字线偏移(X坐标, Y坐标)
    int crosshairOffsetX = buf[33];
    int crosshairOffsetY = buf[34];
    cout<<"十字线偏移(X坐标):"<<crosshairOffsetX<<", (Y坐标):"<<crosshairOffsetY<<endl;

    //偏移像素(X坐标, Y坐标)
    int16_t pixelOffsetX = (buf[36] << 8) | buf[37];
    int16_t pixelOffsetY = (buf[38] << 8) | buf[39];
    cout << "偏移像素(X坐标):" <<pixelOffsetX<<", (Y坐标):"<<pixelOffsetY<<endl;

    //光斑跟踪状态字
    uint8_t spotTrackingStateWord = buf[40];
    if(spotTrackingStateWord == 0x00) cout << "无效值" << endl;
    else if(spotTrackingStateWord == 0x01) cout << "未跟踪到目标" << endl;
    else if(spotTrackingStateWord == 0x02) cout << "跟踪到目标" << endl;
    else if(spotTrackingStateWord == 0x03) cout << "丢失目标" << endl;

    //光斑跟踪(X坐标, Y坐标)
    int16_t lightSpotTrackingX = (buf[42] << 8) | buf[43];
    int16_t lightSpotTrackingY = (buf[44] << 8) | buf[45];
    cout<<"光斑跟踪(X坐标):"<<lightSpotTrackingX<<" (Y坐标):"<<lightSpotTrackingY<<endl;
}

// AI板发给主控422串口数据
void AISend()
{
    const int len = 56;
    uint8_t sendNum = 0x01;
    uint8_t sendTo[58] = {0xEB, 0x90, sendNum, 0x2E, 0x09};
    sendTo[54] = 0x5A;
    sendTo[55] = 0x5A;
    while (1) {
        int64_t ret = obj->serial_send(sendTo, len);         // 串口发送数据
        printf("send data:");
        for (int i = 0; i < ret; i++) {
            printf("[%02X]", sendTo[i]);
        }
        printf("\n");
    }
    if (sendNum == 0xFF) {
        sendNum = 0x01;
    } else {
        sendNum++;
    }
}

// 主控发给AI板422串口数据
void AIRecv()
{
    static uint8_t buf[50]={0xEB,0x90};
    static uint8_t st=0;    //状态机的状态
    static int bi=0;   //消息缓冲区的索引
    static uint32_t msgSum=0;   //成功处理的消息数量
    const int len = 50;
    uint8_t recFrom[len] = {};

    while (1) {
        int64_t ret = obj->serial_recieve(recFrom, len);     // 接收串口数据
        if (ret <= 0) {
            continue;
        }
        if (ret == len) {
            if(recFrom[0]==0xEB && recFrom[1]==0x90) {
                uint16_t crc = MakeCRC_R_LookupShortTable((uint8_t*)(recFrom + 4), len - 6 );
                uint16_t receivedCRC = (recFrom[len - 1] << 8) | recFrom[len - 2];
                // 开始校验
                if (crc == receivedCRC) {
                    printf("recieve data check succeed\nrecieve data:");
                    for (int i = 0; i < ret; i++) {
                        printf("[%02X]", recFrom[i]);
                    }
                    printf("\n");
                    processData(recFrom);
                    ++msgSum;
                } else {
                    printf("recieve data check fail\n data:");
                    for (int i = 0; i < ret; i++) {
                        printf("[%02X]", recFrom[i]);
                    }
                    printf("\n");
                    //continue;
                }
            }
            continue;
        } else {
            for(int i=0; i<ret; ++i) {
                switch(st) {
                    case 0:
                        if(recFrom[i]==0xEB) {
                            st=1;
                            bi=1;
                            //tstamp = createtimestamp();
                        }
                        break;
                    case 1:
                        if(recFrom[i]==0x90) {
                            st=2;
                            bi=2;
                        }
                        else
                            st=0;
                        break;
                    case 2:
                        buf[bi]=recFrom[i];
                        if(bi==49) {
                            st=3;
                        }
                        bi++;
                        break;
                    case 3:
                        uint16_t crc = MakeCRC_R_LookupShortTable((uint8_t*)(buf + 4), len - 6 );
                        uint16_t receivedCRC = ( buf[len - 1] << 8 ) | buf[len - 2];
                        if (crc == receivedCRC) {
                            printf("recieve data check succeed \nrecieve data:");
                            for (int i = 0; i < len; i++) {
                                printf("[%02X]", buf[i]);
                            }
                            printf("\n");
                            processData(recFrom);   //处理数据
                            ++msgSum;
                        } else {
                            printf("recieve data check fail\n data:");
                            for (int i = 0; i < len; i++) {
                                printf("[%02X]", buf[i]);
                            }
                            printf("\n");
                            //continue;
                        }
                        st=0;
                        bi=0;
                        break;
                }
            }
        }
        if (msgSum == 255) {
            msgSum = 1;
        }
    }
}
/***************************串口通信函数结束*******************************/

int IMG_FRAME_SIZE = 640*512;

int main(int argc, char *argv[])
{
    if (argc != 2) {
        ACLLITE_LOG_ERROR("miss exec para, ex: './main localvideoshow/localvideodet/ethvideoshow/ethvideodet'\n");
    }
    string inParam = argv[1];

    if ((inParam != "localvideoshow") &&
        (inParam != "localvideodet") &&
        (inParam != "ethvideoshow") &&
        (inParam != "ethvideodet")) {
        ACLLITE_LOG_ERROR("exec para error, ex: './main localvideoshow/localvideodet/ethvideoshow/ethvideodet'\n");
        return FAILED;
    }

    int sockfd;
    struct sockaddr_in serverAddr, clientAddr;
    uint8_t buffer[1024];
    uint8_t* imgBufRaw;
    bool findHeader = false;
    cv::Mat udpimg = cv::Mat(512,640,CV_8UC1);
    ACLLITE_LOG_INFO("exec %s program\n", inParam.c_str());
    if (inParam == "ethvideoshow" || inParam == "ethvideodet") {

        imgBufRaw = (uint8_t*)malloc(IMG_FRAME_SIZE*sizeof(uint8_t));

        // 创建UDP套接字
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        if (sockfd < 0) {
            std::cerr << "Failed to create socket" << std::endl;
            return FAILED;
        }

        // 设置服务器地址信息
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(10001); // 指定UDP端口
        serverAddr.sin_addr.s_addr = INADDR_ANY;

        // 将套接字绑定到服务器地址
        if (bind(sockfd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0) {
            std::cerr << "Failed to bind socket" << std::endl;
            return FAILED;
        }

        std::cout << "Waiting for data..." << std::endl;

        int nRecvBuf = 5 * 1024 * 1024;
        setsockopt(sockfd,SOL_SOCKET, SO_RCVBUF, (const char *)&nRecvBuf,sizeof(nRecvBuf));
    }
    /**********串口通信初始化**********/
    const char* deviceName = "/dev/ttyUSB0";

    // 创建串口对象
    obj = new Serial(deviceName);

    std::thread aiSend;
    std::thread aiRecv;
    // 初始化串口
    if (obj->init_serial(115200, 8, 'N', 1) < 0) {
        ACLLITE_LOG_ERROR("serial port %s initial fail\n", deviceName);
    } else {
        ACLLITE_LOG_INFO("serial port %s initial success\n", deviceName);
        aiSend = std::thread(AISend);
        aiRecv = std::thread(AIRecv);
        aiSend.detach();
        aiRecv.detach();
    }
    /**********串口通信初始化结束**********/

    // inference
    string fileName;
    bool release = false;
    const char* modelPath = "/det.om";
    const int32_t modelWidth = 640;
    const int32_t modelHeight = 640;
    SampleYOLOV7 sampleYOLO(modelPath, modelWidth, modelHeight);
    Result ret = sampleYOLO.InitResource();
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
        return FAILED;
    }

    std::vector<InferenceOutput> inferOutputs;
    InitUDPSendInfo();

    /*  local video process*/
    if (inParam == "localvideoshow" || inParam == "localvideodet") {
        std::string videopath = "/home/HwHiAiUser/Videos/1.mp4";
        cv::VideoCapture *cvcap = new cv::VideoCapture(videopath);
        while(1) {
            *cvcap>>img;
            if(img.empty()) {
                continue;
            }

            if (inParam != "localvideodet") {
                UdpSendVideo(img);
                continue;
            }
            srcImage = img.clone();
            if (img.rows <= 0) {
                continue;
            }

            cv::resize(img, img,cv::Size(640,640));
            cv::cvtColor(img,img,CV_BGR2YUV_I420);
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
        close(sockfdUDPSend);
        return SUCCESS;
    }

    /*  eth video process*/
    if (inParam != "ethvideoshow" && inParam != "ethvideodet") {
        ACLLITE_LOG_ERROR("current exec program isn't eth video program\n");
        close(sockfdUDPSend);
        return FAILED;
    }

    while(1) {
        int sum = 0;
        auto laststamp = std::chrono::system_clock::now();
        while (true) {
            socklen_t clientAddrLen = sizeof(clientAddr);
            ssize_t dataSize = recvfrom(sockfd, buffer, sizeof(buffer), 0, (struct sockaddr *)&clientAddr, &clientAddrLen);
            if (dataSize < 0) {
                std::cerr << "Failed to receive data" << std::endl;
                return FAILED;
            }

            if(!findHeader) {
                // printf("******in find header*******\n");
                if(IsHeader(buffer)) {
                    ACLLITE_LOG_INFO("******find header*******\n");
                    // printf("sum:%d\n", sum);
                    findHeader = true;
                    continue;
                }
            } else {
                memcpy(imgBufRaw + sum, buffer, dataSize);
                sum += dataSize;

                if(sum == IMG_FRAME_SIZE) {
                    ACLLITE_LOG_INFO("find second header, sum:%d\n", sum);
                    findHeader = false;
                    break;
                }
            }
        }

        if(sum == IMG_FRAME_SIZE) {
            udpimg.data = imgBufRaw;
            auto stamp = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stamp - laststamp);
            laststamp = stamp;
            std::cout<<"time interval:"<< duration.count() << "ms"<<std::endl;

            if (inParam == "ethvideodet") {
                srcImage = udpimg.clone();
                cv::resize(udpimg,img,cv::Size(640,640));
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
            } else {
                UdpSendVideo(udpimg);
            }
            memset(imgBufRaw, 0, IMG_FRAME_SIZE*sizeof(uint8_t));
        } else {
            printf("error img data, sum:%d\n", sum);
        }
    }
    close(sockfdUDPSend);
    return SUCCESS;
}
