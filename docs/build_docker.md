# Build Docker

**Note**:

- run all the following in project root dir.
- run the following one-by-one, or you really know what you're doing.

## Add python3.8/3.9 & pip

目的：在Nvdia提供的基础镜像（`cuda-11.8`）上加入`python3.8` 及 `pip3`

**Output:** ``harbor.lightwheel.net/lwad/env/cuda_python:11.8.0-devel-ubuntu22.04``

```bash
docker build . -f ./docker/cuda_python.dockerfile -t harbor.lightwheel.net/lwad/env/cuda_python:11.8.0-devel-ubuntu22.04
```

## Build torch & necessary packages

目的：在上部分的基础镜像中加入运行lwad必须的大型依赖库，例如`pytorch`

Output: ``harbor.lightwheel.net/lwad/env:{version}``

```bash
docker build . -f ./docker/env.dockerfile -t harbor.lightwheel.net/lwad/env:{version}
```

## Build train env

### 本机配置检验

```bash
cat /etc/docker/daemon.json
```

保证

```json
"default-runtime": "nvidia"
```

被包含。如果修改了配置，重启docker服务：

```bash
sudo systemctl restart docker
```

### 生成含代码的镜像

目的：复制本地代码进入docker内`/workspace`目录，并且安装lwad库

Output: ``harbor.lightwheel.net/lwad/train:1.0.5``

Assume the version of train env is ``1.0.5``

```bash
bash build_docker.sh 1.0.5
```
