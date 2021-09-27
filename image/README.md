<!--
 * @Author: LawsonAbs
 * @Date: 2021-09-25 18:37:31
 * @LastEditTime: 2021-09-27 08:04:48
 * @FilePath: /daguan_gitee/image/README.md
-->
# 0. 环境配置
- Docker version 19.03.12
- NVIDIA Driver Version: 455.45.01
- CUDA Version: 11.1

因为百度网盘限制，无法一次性上传超过4G大小的文件，所以使用linux中的split分割得到的小文件，需拼接使用。
百度网盘链接地址：
链接：https://pan.baidu.com/s/1ElbULCc0vhr2neeXsXXQlw 
提取码：pbv8
百度网盘有效期是30天，请于2021/10/27号之前完成下载。
# 1. 复现操作
按照如下操作，即可完全复现B榜提交结果。
```
## 拼接
cat daguan* > daguan_test.tar
## 加载镜像
docker load < daguan_test.tar
## 起容器

如果使用挂载参数 -v /data:/data， 请确保先把user_data 下的模型copy出来再覆盖！
sudo docker run --gpus all -itd --name daguan_prod  daguan_test:2 
sh run.sh
or
sudo nvidia-docker run -itd --name daguan_prod daguan_test:2 
sh run.sh
```
等待程序运行完毕，进入data/prediction_result 下查看 result.txt 文件