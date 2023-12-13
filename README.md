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