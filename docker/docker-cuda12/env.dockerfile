FROM harbor.lightwheel.net/lwad/env/pytorch:24.10-py3

USER root
WORKDIR /workspace
ENV PATH="/root/.local/bin:/usr/bin:$PATH"
# ENV CUDA_HOME=/usr/local/cuda-11.8  # included in pytorch:24.10-py3

# for python 3.10
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt install curl zsh sudo wget python3-full net-tools -y
RUN apt install vim nano libegl1 git git-lfs psmisc htop nvtop ffmpeg net-tools lsof openssh-server dstat iotop -y 

RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
RUN pip install ninja cython addict packaging Pillow matplotlib prettytable six terminaltables lyft_dataset_sdk scikit-image tensorboard cityscapesscripts imagecorruptions scipy scikit-learn networkx ipython opencv-python==4.8.0.74 seaborn einops casadi torchmetrics trimesh pytest pytest-cov pytest-runner flake8 similaritymeasures py-trees simple_watchdog_timer transforms3d tabulate ephem dictor open3d 

# Specific versions and conditions
RUN pip install numba motmetrics yapf==0.40.1 trimesh laspy lazrs torch_scatter deepspeed

# Platform-specific packages
RUN pip install "regex; sys_platform=='win32'" "pycocotools; platform_system == 'Linux'" "pycocotools-windows; platform_system == 'Windows'" 


RUN https_proxy=http://10.10.0.78:7890 http_proxy=http://10.10.0.78:7890 all_proxy=socks5://10.10.0.78:7890 sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Set Zsh as the default shell
RUN chsh -s $(which zsh)

# Install zsh-completions, zsh-syntax-highlighting, and zsh-autosuggestions
RUN https_proxy=http://10.10.0.78:7890 http_proxy=http://10.10.0.78:7890 all_proxy=socks5://10.10.0.78:7890 git clone https://github.com/zsh-users/zsh-completions ~/.oh-my-zsh/custom/plugins/zsh-completions && \
    https_proxy=http://10.10.0.78:7890 http_proxy=http://10.10.0.78:7890 all_proxy=socks5://10.10.0.78:7890 git clone https://github.com/zsh-users/zsh-syntax-highlighting ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting && \
    https_proxy=http://10.10.0.78:7890 http_proxy=http://10.10.0.78:7890 all_proxy=socks5://10.10.0.78:7890 git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions

# Configure Zsh plugins and settings in .zshrc
RUN echo "plugins=(git zsh-completions zsh-syntax-highlighting zsh-autosuggestions)" >> ~/.zshrc && \
    echo "source ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ~/.zshrc && \
    echo "source ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh" >> ~/.zshrc && \
    echo "setopt CORRECT" >> ~/.zshrc && \
    echo "setopt CORRECT_ALL" >> ~/.zshrc

# Set working directory and default entry point to zsh
RUN echo 'export ZSH_DISABLE_COMPFIX="true"' >> ~/.zshrc

# initialzie ssh
RUN mkdir /var/run/sshd
RUN echo 'root:lw' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config

# ssh copy key
RUN ssh-keygen -t rsa -b 4096 -C "lwad" -N "" -f /root/.ssh/id_rsa
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys

# create a script to handle SSH startup
RUN printf '#!/bin/bash\n/usr/sbin/sshd' > /opt/nvidia/entrypoint.d/80-start-ssh.sh 

EXPOSE 22