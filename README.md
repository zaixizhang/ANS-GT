# ANS-GT
Pytorch implementation of NeurIPS'22 paper "Hierarchical Graph Transformer with Adaptive Node Sampling"(https://arxiv.org/abs/2210.03930)

The preliminary version of our code: https://github.com/zaixizhang/Graph_Transformer


### Python environment setup with Conda

```bash
conda create -n gt python=3.9
conda activate gt

conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge

pip install ogb
pip install pygsp
pip install scipy

conda clean --all
```


### Running the Code
```bash
conda activate gt
python3 preprocess_data.py
sh start.sh
```

## Citation

If you find this work useful, please cite our paper:
```bibtex
@article{zhang2022hierarchical,
  title={Hierarchical Graph Transformer with Adaptive Node Sampling},
  author={Zhang, Zaixi and Liu, Qi and Hu, Qingyong and Lee, Chee-Kong},
  journal={arXiv preprint arXiv:2210.03930},
  year={2022}
}
```
