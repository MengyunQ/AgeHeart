# AgeHeart
paperlink: https://arxiv.org/abs/2208.13146

Generative model for aging cardiac segmentation
data preprocessing:
prepare the dataset into labelmap.
## Train

```python
python Train_AgeHeart.py --age_loss_weight 1e-1 --dit_loss_weight 1e-1 --cyc_loss_weight 1e-1 --mapping
```


## Citations

```bibtex
@article{qiao2022generative,
  title={Generative Modelling of the Ageing Heart with Cross-Sectional Imaging and Clinical Data},
  author={Qiao, Mengyun and Basaran, Berke Doga and Qiu, Huaqi and Wang, Shuo and Guo, Yi and Wang, Yuanyuan and Matthews, Paul M and Rueckert, Daniel and Bai, Wenjia},
  journal={arXiv preprint arXiv:2208.13146},
  year={2022}
}
```
