# 光轮端到端系统

## 相关文档
- [环境配置文档](docs/env_install.md)
- [快系统数据详解](docs/vad_data.md)
- [可视化工具](docs/viz_tools.md)
- [Docker配置](docs/build_docker.md)
- [Deepspeed多机](docs/deepspeed.md)
- [分场景开环评测](docs/scenario_test.md)
- [开环评测指标说明](docs/evaluation_metrics.md)
- [慢系统数据预刷](docs/vlmdatagen.md)

## 项目交付文档
- [光轮端到端快系统代码交付及使用指南](docs/release/光轮端到端快系统代码交付及使用指南.pdf)
- [光轮端到端模型交付报告](docs/release/光轮端到端模型交付报告.pdf)
- [2024-10-30](docs/release/2024-10-30.pdf)

## 配置docker环境

### 拉取最新的docker env

```bash
docker pull harbor.lightwheel.net/lwad/env:2.1.0
```

如果遇到`Error response from daemon: unauthorized: unauthorized to access repository: lwad/env, action: pull: unauthorized to access repository: lwad/env, action: pull`，请使用

```bash
docker login harbor.lightwheel.net
sudo systemctl restart docker
```

登录。还不行的话联系`Jay Yang`。

### 启动docker，映射本地目录、gpu设备等

以下命令把``/home/{your_username}/``挂载到docker内的``/root/``目录。

```bash
docker run -it \
--name lwad \
--user root \
-P \
--shm-size 64G --ulimit memlock=-1 --ulimit stack=67108864 \
--runtime=nvidia  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --gpus all \
-v /data/ceph:/data/ceph \
-v /data/ceph/homes/$(whoami)/:/workspace/ \
harbor.lightwheel.net/lwad/env:latest
```

注：自动启动ssh服务，`root`账户的密码是`lw`。 在启动docker的机器上运行

```bash
docker port lwad
```

可以看到类似`22/tcp -> 0.0.0.0:32771` 的输出。在这里`32771`即为docker端口。我们可以从别的机器通过

```bash 
ssh root@172.16.100.252 -p 32771
``` 

(`172.16.100.252`为宿主机id) 直接登陆docker, 默认密码`lw`。

### 在docker内部重新安装lwad

```bash
cd /root/{path_2_your_lwad}/lwad && python3.8 -m pip install -v -e .
```


## 训练 & 评测脚本

### 训练脚本
注意，以下代码运行在`lwad`目录内

- **方法一： 在`.vscode/launch.json`中配置debugger**

```json
{
    "name": "lightwheel VAD train",
    "type": "debugpy",
    "request": "launch",
    "module": "torch.distributed.launch",
    "args": [
        "--nproc_per_node=8",
        "--master_port=28509",
        "./adbase/vad/train.py",
        "./adbase/vad/configs/VAD/VAD_lightwheel_config.py",
        "--launcher=pytorch",
        "--deterministic"
    ], 
}
```

主要调整 `--nproc_per_node`： 节点上的GPU数量

- **方法二：命令行**

```shell
torchrun \
    --nproc_per_node=4 \
    --master_port=28509 \
    ./adbase/vad/train.py \
    ./adbase/vad/configs/VAD/VAD_lightwheel_config.py \
    --launcher=pytorch \
    --deterministic
```

####  模型存储及训练监控

模型默认存储在 `lwad/work_dirs` 文件夹，每epoch一次进行存储，如果需要调整，修改配置文件中的`463`行附近：

```python
checkpoint_config = dict(interval=1000, by_epoch=False, max_keep_ckpts=total_epochs)
```

使用tensorboard 可视化训练。注意，默认环境中的tensorboard版本较低，建议新建环境后安装tensorboard并运行

```bash
tensorboard --logdir work_dirs/
```

### 测试脚本

#### 生成评测真值地图数据

为了正确地跑出地图评测的结果，需要先提取出测试集的真值地图信息，即从pkl格式的转为json格式的真值地图信息，参考如下命令：

```bash
python convert_gt_map_json.py \
    --data_root Path/to/nuplan \ 
    --pkl_path  Path/to/nuplan/test/pkl \
    --save_path eval_map.json
```

#### 模型评测

以下命令用于运行测试脚本：

- **方法一： 在`.vscode/launch.json` 中配置debugger**

```json
{
    "name": "lightwheel VAD test",
    "type": "debugpy",
    "request": "launch",
    "module": "torch.distributed.launch", 
    "args": [
        "--nproc_per_node=2", 
        "--master_port=29303", 
        "./adbase/vad/test.py", 
        "./adbase/vad/configs/VAD/VAD_lightwheel_config.py",
        "path/to/model.pth",
        "--json_dir=../test/"
        "--launcher=pytorch",
        "--eval=bbox"
    ]
}
```

- **方法二：命令行**

```bash
torchrun \
    --nproc_per_node=2 \
    --master_port=29303 \
    ./adbase/vad/test.py ./adbase/vad/configs/VAD/VAD_lightwheel_config.py \
    /data/ckpts/H800_1015_epoch9.pth \
    --json_dir='directory_path_to_save_result_json'
    --launcher=pytorch \
    --eval=bbox
```

## Docker 发版及使用

### 压缩

```bash
docker save harbor.lightwheel.net/lwad/train:2.0.0 | pv | lbzip2 > train_1.0.0.tar.bz2
```

### 解压

```bash
cat train_2.0.0.tar.bz | pv | lbzip2 -d | docker load
```

- 使用

`-v /data/ceph:/data/ceph` 映射数据盘

```bash
docker run -it \
    --name lwad \
    --user root \
    -P \
    --shm-size 64G --ulimit memlock=-1 --ulimit stack=67108864 \
    --runtime=nvidia  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all --gpus all \
    -v /data/ceph:/data/ceph \
    harbor.lightwheel.net/lwad/train:2.0.0
```

默认定向到`/workspace`
