# deepspeed多机多卡训练

DeepSpeed API是PyTorch的一个轻量级封装，可以实现分布式的多机多卡训练。

## docker容器下的deepspeed多机训练

多机使用docker容器部署环境，具体请参考[README](../README.md)中关于docker训练的说明。

举例来讲，假设在服务器A和服务器B上各部署一个容器，容器名分别为`lwad`和`lwad2`：

在服务器A上（假设IP为172.16.100.250）：

```bash
docker run -it \
--name lwad \
--user root \
-P \
--shm-size 64G --ulimit memlock=-1 --ulimit stack=67108864 \
--runtime=nvidia  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --gpus all \
-v /data/ceph:/data/ceph \
-v /data/ceph/homes/$(whoami)/:/root/ \
harbor.lightwheel.net/lwad/env:latest
```

并通过`cd ~/lwad/lwad && python3.8 -m pip install -v -e .`重新安装lwad。

在服务器B上（假设IP为172.16.100.248）：

```bash
docker run -it \
--name lwad2 \
--user root \
-P \
--shm-size 64G --ulimit memlock=-1 --ulimit stack=67108864 \
--runtime=nvidia  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --gpus all \
-v /data/ceph:/data/ceph \
-v /data/ceph/homes/$(whoami)/:/root/ \
harbor.lightwheel.net/lwad/env:latest
```

并通过`cd ~/lwad/lwad && python3.8 -m pip install -v -e .`重新安装lwad。

在每台机器上使用`docker port`命令查看lwad容器对应的端口，并配置`ds_config/hostfile.txt`

例：假设在机器A和机器B上，建立名为`lwad`和`lwad2`的两个容器，则可以使用`docker port`命令查看lwad容器端口。

```bash
# 在服务器A上查看lwad容器端口
docker port lwad

# 输出示例
22/tcp -> 0.0.0.0:32769
22/tcp -> [::]:32769

# 在服务器B上查看lwad_2容器端口
docker port lwad2

# 输出示例
22/tcp -> 0.0.0.0:32768
22/tcp -> [::]:32768
```

则`hostfile.txt`配置如下，代表两台机器，每台机器上各使用一张GPU卡，lwad使用32769端口和GPU0和GPU1，lwad2使用32768端口和GPU0和GPU1。

```plaintext
172.16.100.250:32769 slots=2 gpu=0,1
172.16.100.248:32768 slots=2 gpu=0,1
```

接下来在`~/lwad/lwad/`目录里运行以下命令：

```bash
deepspeed --hostfile deepspeed_cfg/hostfile.txt \
          --master_addr 172.16.100.250:32769 \
          --master_port 29600 \
          ./adbase/vad/train.py \
          ./adbase/vad/configs/VAD/VAD_lightwheel_config.py \
          --launcher pytorch \
          --deterministic \
          --deepspeed \
          --deepspeed_config ./ds_config/ds_config.json
```

在命令中指定`--deepspeed`和`--deepspeed_config`，分别指定使用deepspeed和deepspeed配置文件。

`deepspeed_config`是deepspeed的配置文件，以下为一个示例：

```json
{
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 1,
    "steps_per_print": 2000
}
```

- `train_batch_size`指定训练的批量大小。
- `train_micro_batch_size_per_gpu`指定每个GPU卡上的mini batch大小。

需要注意的是，`train_batch_size`必须是`train_micro_batch_size_per_gpu`的整数倍。

即可成功跑起多机训练。

更多关于deepspeed配置文档的详解，请参考[deepspeed配置文档](https://www.deepspeed.ai/docs/config-json/)。

