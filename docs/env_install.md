# lwad 环境配置文档


## CUDA

确认`cuda-11.8`已安装

```bash
ls /usr/local
```
如果没有`cuda-11.8`，前往 [NVIDIA 官方 CUDA 下载页面](https://developer.nvidia.com/cuda-toolkit-archive) 下载 CUDA 11.8。选择合适的操作系统和安装方式（通常为`.run`文件或`.deb`包）。以`.run`为例：

```bash
sudo chmod +x cuda_11.8.*_linux.run # 赋予.run 文件可执行权限
sudo ./cuda_11.8.*_linux.run # 运行安装程序
```

### 配置环境变量

在 `/usr/local` 下安装完成后，你需要将 CUDA 相关的路径添加到系统的环境变量中。

1. 编辑 `.bashrc` 文件（或者 `.zshrc` 文件，如果你使用 Zsh）：

   ```bash
   nano ~/.bashrc
   ```
   
2. 在文件末尾添加以下内容：

   ```bash
   export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
   export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
   ```

3. 保存并退出编辑器，接着通过以下命令加载新的环境变量：

   ```bash
   source ~/.bashrc
   ```

## 新建虚拟环境

```bash
mkvirtualenv lwad  --python=python3.8
```

## 设置镜像源获得最佳下载速度

```
pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
```

## 安装依赖库
**注：Torch要求cuda-11.8**

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install ninja packaging
```

## 配置环境变量

```bash
export PATH=/usr/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
```

## 安装lwad库

注意，必须在含``cuda-11.8``本机编译mmcv, 否则会报 ``error in ms_deformable_im2col_cuda: no kernal image is available for execution on the device`` 

```bash
cd lwad && python -m pip install -v -e .
```

## 下载预训练模型

注：如果`lwad/ckpts`中已经存在，跳过此步

```bash
cd lwad/ckpts
wget https://hf-mirror.com/rethinklab/Bench2DriveZoo/resolve/main/resnet50-19c8e357.pth
wget https://hf-mirror.com/rethinklab/Bench2DriveZoo/resolve/main/r101_dcn_fcos3d_pretrain.pth
```
