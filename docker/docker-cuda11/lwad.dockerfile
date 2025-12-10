FROM harbor.lightwheel.net/lwad/env:cuda11-3.1.0a1

RUN mkdir -p /workspace/lwad
COPY tools /workspace/lwad/tools
RUN cd /workspace/lwad/tools/nuplan-devkit && python3 -m pip install -e . && pip install -r requirements_torch.txt && pip install -r requirements.txt && pip cache purge

# install lwad
COPY ckpts /workspace/lwad/ckpts
COPY lwad_cp /workspace/lwad/lwad

ENV PATH="$CUDA_HOME/bin:$PATH"
ENV OMP_NUM_THREADS=1
RUN cd /workspace/lwad/lwad && pip install -v -e . && pip cache purge
RUN pip install opencv-python==4.8.0.74 && pip cache purge