# 2024_ICML_TSS
Offical implementation for Mitigating label noise on graph via topological sample selection (ICML 2024) (PyTorch implementation).  

This is the code for the paper:
[Mitigating Label Noise on Graph via Topological Sample Selection](https://arxiv.org/abs/2403.01942)      
Yuhao Wu, Jiangchao Yao, Xiaobo Xia, Jun Yu, Ruxin Wang, Bo Han, Tongliang Liu.

If you find this code useful in your research, please cite  
```bash
@inproceedings{wumitigating,
  title={Mitigating Label Noise on Graphs via Topological Sample Selection},
  author={Wu, Yuhao and Yao, Jiangchao and Xia, Xiaobo and Yu, Jun and Wang, Ruxin and Han, Bo and Liu, Tongliang},
  booktitle={Forty-first International Conference on Machine Learning}
}
```  
## Dependencies
we implement our methods by PyTorch on NVIDIA RTX 4090. The environment is as bellow:
- [PyTorch](https://PyTorch.org/), version == 2.2.2
- [CUDA](https://developer.nvidia.com/cuda-downloads), version == 11.8

## Experiments     
Here is an example: 
```bash
python main.py --dataset cora --noise_rate 0.5
```
