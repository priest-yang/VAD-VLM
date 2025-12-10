#!/usr/bin/env bash

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <tag> <cuda_version>"
    echo "Example: $0 v1.0 cuda11"
    exit 1
fi

TAG=$1
CUDA_VERSION=$2

# Validate CUDA version
if [ "$CUDA_VERSION" != "cuda11" ] && [ "$CUDA_VERSION" != "cuda12" ]; then
    echo "Error: CUDA version must be either 'cuda11' or 'cuda12'"
    exit 1
fi

set -euxo pipefail
rm -rf lwad_cp
cp -rl lwad lwad_cp
rm -rf lwad_cp/.git
rm lwad_cp/adbase/vad/configs/VAD/VAD_base_e2e_*
docker build . -f "./docker/docker-${CUDA_VERSION}/lwad.dockerfile" -t harbor.lightwheel.net/lwad/train:${TAG}
rm -rf lwad_cp/


# Usage
# # For CUDA 11
# ./build_docker.sh v1.0 cuda11

# # For CUDA 12
# ./build_docker.sh v1.0 cuda12