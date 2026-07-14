#基础配置
export PYTORCH_ENABLE_SAME_RAND_A100=1
export MACA_SMALL_PAGESIZE_ENABLE=1
export SET_DEVICE_NUMA_PREFERRED=1
export PYTORCH_USE_FLASHATTN=1
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0

# setup MACA path cu-bridge
export MACA_PATH="/opt/maca"
export MACA_CLANG=${MACA_PATH}/mxgpu_llvm
export CUDA_PATH=${MACA_PATH}/tools/cu-bridge
export PATH=${CUDA_PATH}:${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_CLANG}/lib:${LD_LIBRARY_PATH:-}

export CUCC_CMAKE_ENTRY=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MALLOC_THRESHOLD=99
#网络配置
export MCCL_P2P_LEVEL=SYS
export MCCL_FAST_WRITE_BACK=1
export MCCL_EARLY_WRITE_BACK=15
export MCCL_NET_GDR_LEVEL=SYS
export MCCL_CROSS_NIC=1
export MCCL_ENABLE_FC=1
export MCCL_LIMIT_RING_LL_THREADTHRESHOLDS=1
export MCCL_MAX_NCHANNELS=16
export OMP_NUM_THREADS=16
export VLLM_NCCL_SO_PATH=${MACA_PATH}/lib/libmccl.so


# export MCCL_SHM_DISABLE=0 #设置为1后将禁用共享内存（SHM）传输 MCCL将使用网络（即InfiniBand或IP套接字）在CPU套接字之间进行通信。
# export MCCL_DEBUG=INFO
# export MCCL_P2P_DISABLE=1 #禁用基于PCIe或MetaXLink的点对点（P2P）传输。
export MACA_MPS_MODE=1
# unset VLLM_NCCL_SO_PATH
# solve no nccl timeout in compiling fused kernortels wait
# MCCL_ASYNC_ERROR_HANDLING=3 会在 NCCL 异常时直接 SIGKILL 进程 → exit 137
# 改用 PyTorch 层的 TORCH_NCCL_ASYNC_ERROR_HANDLING，容错更温和
unset NCCL_ASYNC_ERROR_HANDLING
unset MCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MCCL_SOCKET_IFNAME=bond0 #2/4卡开发环境：eth0 多卡：bond0 具体查一下ifconfig看有没有bond0
export GLOO_SOCKET_IFNAME=bond0
export MCCL_IB_HCA=mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5
export MACA_DIRECT_DISPATCH=1
export FLAGS_set_to_1d=False
#export FLAGS_dataloader_use_file_descriptor=False
export FLAGS_embedding_deterministic=1

export MAX_JOBS=4
export MACA_NUM_COMPILE_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export HYDRA_FULL_ERROR=1
export MCPYTORCH_DISABLE_PRINT=1
#export VLLM_USE_V1=0
unset PAGEABLE_MEMCPY_ASYNC
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
