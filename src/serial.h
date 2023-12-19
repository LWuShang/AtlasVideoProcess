#ifndef SERIAL_H
#define SERIAL_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

#include <termios.h> //set baud rate

#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/stat.h>

//自检指令  1有效 0无效
struct SelfCheckInstruction {
    bool PowerOnSelfCheck;                 //上电自检
    bool PeriodicSelfCheck;                //周期自检
    bool ReadProductIdentifier;            //读取产品标识码
    SelfCheckInstruction():PowerOnSelfCheck(false), PeriodicSelfCheck(false), ReadProductIdentifier(false){}
};

//工作模式指令
enum class ModeControlInstruction {
    InvalidMode = 0x00,                    //无效
    FollowSearch = 0x01,                   //随动搜索
    AutoSearch = 0x02,                     //自动搜索
    ManualSearch = 0x03,                   //手动搜索
    SnowPlowSearch = 0x04,                 //雪犁搜索
    VideoTracking = 0x05,                  //视频跟踪
    LightSpotSearch = 0x06,                //光斑搜索
    LightSpotTracking = 0x07,              //光斑跟踪
    GeographicTrackingMode = 0x08,         //地理跟踪模式
    MapMatching = 0x09                     //地图匹配
};

//发送模板
enum class SendTemplateInstruction {
    InvalidMode = 0x00,                    //无效
    SendTemplate = 0x01                    //发送目标模板
};

class Serial
{
public:
    //显示构造
    explicit Serial(const char* devicename);
    ~Serial();

    //打开并初始化设备，
    //入参：无
    //出参：无
    //返回值：打开的串口fd，-1-打开设备失败
    const int init_serial(const int nSpeed, const int nBits, const char nEvent, const int nStop); //波特率，数据位，奇偶校验，停止位

    //向设备发送数据
    //入参：发送数据缓冲区、发送的数据量
    //出参：无
    //返回值：发送的数据量
    const uint64_t serial_send(const uint8_t* buffSenData, const uint64_t sendLen);    //buffSenData should len 1024

    //从设备接收数据
    //入参：接收数据缓冲区
    //出参：无
    //返回值：接收的数据量，-1表示出错
    const int64_t serial_recieve(uint8_t* buffRcvData, const uint64_t recvLen);

    //关闭串口
    //返回值：0-succeed，-1-fail
    const int close_serial();

    //禁用拷贝构造函数
    //Serial(const Serial& obj) delete;
    //禁用复制构造函数
    //Serial& operator&=(const Serial& obj) delete;


private:
    //根据设备名称打开设备
    //入参：设备名称
    //返回值：打开的串口fd，-1-打开设备失败
    const int openPort();

    //设置设备参数
    //入参：nSpeed-波特率，nBits-数据位，nEvent-校验方式，nStop-停止位
    //返回值：0-succeed，-1-fail
    const int setOpt(const int nSpeed, const int nBits, const char nEvent, const int nStop);

    //读取设备数据
    //入参：rcv_buf-读取数据存储缓冲区，TimeOut-读取数据超时时间(单位ms)，Len-本次需要读取的数据量
    //
    //返回值：本次实际读取的数据量，-1表示出错
    const int64_t readDataTty(uint8_t* rcv_buf, const int timeOut, const uint64_t len);

    //发送设备数据
    //入参：send_buf-发送数据缓冲区，TimeOut-发送数据超时时间(单位ms)，Len-发送数据量
    //返回值：发送的数据量
    const uint64_t sendDataTty(const uint8_t* send_buf, const int timeOut, const uint64_t len);

private:
    char* m_device;//串口设备
    int m_fd;//打开的设备端口
    int m_setOpt;//SetOpt 的增量i
};

#endif