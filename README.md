# AtlasVideoProcess
Processing videos based on Atlas development board

初始化配置需要参考URL：https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleYOLOV7

单独配置yaml-cpp:
下载yaml-cpp源码
```
git clone https://github.com/jbeder/yaml-cpp.git
```
编译安装yaml-cpp
```
cd yaml-cpp
mkdir build 
cd build
cmake ..
make
make install
```
查看yaml-cpp是否安装成功
```
ll /usr/local/lib/libyaml-cpp.a

```
若存在则表示安装成功

6.安装KCFcpp-master
#下载KCFcpp-mster源码
```
https://github.com/LeRoii/robusttracker/tree/main/KCFcpp-master
(直接下载KCFcpp-mster压缩包）
```
编译安装KCFcpp-master
```
cd KCFcpp-master
mkdir build
cd build
cmake ..
sudo make
make install
```