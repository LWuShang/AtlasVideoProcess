#include <iostream>
#include "serial.h"

Serial::Serial(const char* devicename):
    m_fd(-1),m_setOpt(0)
{
    if(devicename != NULL && devicename != nullptr) {
        this->m_device = new char[strlen(devicename)+1];
        strcpy(this->m_device, devicename);  //拷贝内存，目标，数据源，字节数
    }
}

Serial::~Serial()
{
    //关闭串口
    if(this->m_fd > 0) {
        close_serial();
    }
    //释放缓存
    if(this->m_device != NULL && this->m_device != nullptr) {
        delete [] this->m_device;
    }
}


/*
 * 打开并初始化设备，
 * 返回值：打开的串口fd，-1-打开设备失败
 */
const int Serial::init_serial(const int nSpeed, const int nBits, const char nEvent, const int nStop)
{
    if (this->m_device == NULL || this->m_device == nullptr) {
        return -1;
    }

    //打开设备
    this->m_fd = open(this->m_device, O_RDWR | O_NOCTTY | O_NDELAY);
    if (this->m_fd == -1) {
        printf("Can't Open Serial Port :%s\n", this->m_device);
        return (-1);
    } else {
        printf("open %s succeed .....\n", this->m_device);
    }

    if (fcntl(this->m_fd, F_SETFL, 0)<0) {  //锁？ F_SETFL ：改变open设置的标志
        printf("fcntl failed!\n");  //锁失败？
    } else {
        printf("fcntl=%d\n", fcntl(this->m_fd, F_SETFL, 0));
    }

    if (isatty(STDIN_FILENO) == 0) {  //isatty判断文件描述符是否是终端设备
        printf("standard input is not a terminal device\n");  //标准输入不是终端设备
    } else {
        printf("is a tty success!\n");
    }
    printf("fd-open=%d\n", this->m_fd);

    //初始化设备
    if(setOpt(nSpeed, nBits, nEvent, nStop) < 0) {
        //初始化设备失败，关闭设备
        close_serial();
    }
    return this->m_fd;
}


//设置设备参数
//入参：nSpeed-波特率，nBits-数据位，nEvent-校验方式，nStop-停止位
//返回值：0-succeed，-1-fail
const int Serial::setOpt(const int nSpeed, const int nBits, const char nEvent, const int nStop)
{
    if(this->m_fd < 0) {
        printf("Don`t set serial options where serial port not opened\n");
        return -1;
    }
    struct termios newtio, oldtio;
    if (tcgetattr(this->m_fd, &oldtio) != 0) {
        perror("SetupSerial  tcgetatt execute fail\n");
        return -1;
    }
    bzero(&newtio, sizeof(newtio));
    newtio.c_cflag |= CLOCAL | CREAD;
    newtio.c_cflag &= ~CSIZE;

    switch (nBits) {
        case 7:
            newtio.c_cflag |= CS7;
            break;
        case 8:
            newtio.c_cflag |= CS8;
            break;
    }

    switch (nEvent) {
        case 'O':                     //奇校验
            newtio.c_cflag |= PARENB;
            newtio.c_cflag |= PARODD;
            newtio.c_iflag |= (INPCK | ISTRIP);
            break;
        case 'E':                     //偶校验
            newtio.c_iflag |= (INPCK | ISTRIP);
            newtio.c_cflag |= PARENB;
            newtio.c_cflag &= ~PARODD;
            break;
        case 'N':                    //无校验
            newtio.c_cflag &= ~PARENB;
            break;
    }

    switch (nSpeed) {
        case 2400:
            cfsetispeed(&newtio, B2400);
            cfsetospeed(&newtio, B2400);
            break;
        case 4800:
            cfsetispeed(&newtio, B4800);
            cfsetospeed(&newtio, B4800);
            break;
        case 9600:
            cfsetispeed(&newtio, B9600);
            cfsetospeed(&newtio, B9600);
            break;
        case 115200:
            cfsetispeed(&newtio, B115200);
            cfsetospeed(&newtio, B115200);
            break;
        case 921600:
            cfsetispeed(&newtio, B921600);
            cfsetospeed(&newtio, B921600);
            break;
        default:
            cfsetispeed(&newtio, B9600);
            cfsetospeed(&newtio, B9600);
            break;
    }
    if (nStop == 1) {
        newtio.c_cflag &= ~CSTOPB;
    } else if (nStop == 2) {
        newtio.c_cflag |= CSTOPB;
    }
    newtio.c_cc[VTIME] = 0;
    newtio.c_cc[VMIN] = 0;
    tcflush(this->m_fd, TCIFLUSH);
    if ((tcsetattr(this->m_fd, TCSANOW, &newtio)) != 0) {
        perror("serial port set error where tcsetattr execute fail\n");
        return -1;
    }
    printf("serial port set succeed!\n");
    return 0;
}


//读取设备数据
//入参：rcv_buf-读取数据存储缓冲区，TimeOut-读取数据超时时间(单位ms)，Len-本次需要读取的数据量
//返回值：本次实际读取的数据量，-1表示出错
const int64_t Serial::readDataTty(uint8_t* rcv_buf, const int timeOut,const uint64_t len)
{
    if(this->m_fd < 0) {
        printf("Don`t readData from serial port where serial port not opened\n");
        return -1;
    }

    int retval;
    fd_set rfds;
    struct timeval tv;
    int ret;
    tv.tv_sec = timeOut / 1000;  //set the rcv wait time
    tv.tv_usec = timeOut % 1000 * 1000;  //100000us = 0.1s

    //本次读取的数据总量
    int64_t pos = 0;

    //需要读取的数据总量 大于 当前实际已读取的数据总量，则继续读取数据
    while (len > pos) {
        FD_ZERO(&rfds);
        FD_SET(this->m_fd, &rfds);
        retval = select(this->m_fd + 1, &rfds, NULL, NULL, &tv);  //select返回值大于零就是描述字的数目，-1出错，0超时
        if (retval == -1) {
            perror("select fail where on execute readDataTty\n");
            break;
        } else if (retval) {
            //读取数据
            ret = read(this->m_fd, rcv_buf + pos, len-pos);
            if (-1 == ret) {
                break;
            }

            pos += ret;
        } else {
            break;
        }
    }

    return pos;
}

/*
 * 发送设备数据
 * 入参：send_buf-发送数据缓冲区，TimeOut-发送数据超时时间(单位ms)，Len-发送数据量
 * 返回值：发送的数据量
 */
const uint64_t Serial::sendDataTty(const uint8_t* send_buf, const int timeOut, const uint64_t len)
{
    if(this->m_fd < 0) {
        printf("Don`t sendData to serial port where serial port not opened\n");
        return -1;
    }

    ssize_t ret;

    int retval;
    fd_set wfds;
    struct timeval tv;
    tv.tv_sec = timeOut / 1000;  //set the rcv wait time
    tv.tv_usec = timeOut % 1000 * 1000;  //100000us = 0.1s

    //本次读取的数据总量
    uint64_t pos = 0;

    //需要发送的数据总量 大于 当前实际已发送的数据总量，则继续发送数据
    while (len > pos) {
        FD_ZERO(&wfds);
        FD_SET(this->m_fd, &wfds);
        retval = select(this->m_fd + 1, NULL, &wfds, NULL, &tv);
        if (retval == -1) {
            perror("select fail where on execute readDataTty\n");
            break;
        } else if (retval) {
            //读取数据
            ret = write(this->m_fd, send_buf + pos, len-pos);
            if (-1 == ret) {
                break;
            }

            pos += ret;
        } else {
            break;
        }
    }

    return pos;
}


//向设备发送数据
//入参：发送数据缓冲区、发送的数据量
//出参：无
//返回值：发送的数据量
const uint64_t Serial::serial_send(const uint8_t* buffSenData, const uint64_t sendLen)
{
    return sendDataTty(buffSenData, 2 ,sendLen);
}

//从设备接收数据
//入参：接收数据缓冲区
//出参：无
//返回值：接收的数据量，-1表示出错
const int64_t Serial::serial_recieve(uint8_t* buffRcvData, const uint64_t recvLen)
{
    return readDataTty(buffRcvData, 2, recvLen);
}

//关闭串口
//返回值：0-succeed，-1-fail
const int Serial::close_serial()
{
    if(this->m_fd > 0) {
        close(this->m_fd);
        this->m_fd = -1;
    }
    return 0;
}