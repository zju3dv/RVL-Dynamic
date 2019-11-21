# Prior Guided Dropout for Robust Visual Localization in Dynamic Environments
> [Prior Guided Dropout for Robust Visual Localization in Dynamic Environments](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_Prior_Guided_Dropout_for_Robust_Visual_Localization_in_Dynamic_Environments_ICCV_2019_paper.pdf)  
>Zhaoyang Huang, Yan Xu, Jianping Shi, Xiaowei Zhou, Hujun Bao, Guofeng Zhang  
>The code will be released soon (perhaps on the end of November).

## License
Licensed under the CC BY-NC-SA 4.0 license, see [LICENSE](LICENSE.md). 

## Citation
If you find this code useful for your research, please cite our paper

```
@inproceedings{huang2019prior,
  title={Prior Guided Dropout for Robust Visual Localization in Dynamic Environments},
  author={Huang, Zhaoyang and Xu, Yan and Shi, Jianping and Zhou, Xiaowei and Bao, Hujun and Zhang, Guofeng},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2791--2800},
  year={2019}
}
```
## Environment
PGD-MapNet uses Conda to setup the environment
```
conda env create -f environment.yml
conda activate pgd-mapnet
```
The data is processed as suggested in geomapnet.
The dynamic information computed from Mask_RCNN is stored in `datainfo`.
The files should be put into the corresponding root dir of each scene.
## Running
### Training
```
cd experiments
bash runattmapnet.sh
```
### Evaluation
```
cp logs/exp_beta[-3.0]gamma[-3.0]batch_size[64]model[attentionmapnet]mask_sampling[True]sampling_threshold[0.2]color_jitter[0.0]uniform_sampling[False]mask_image[False]dataset[RobotCar]scene[full]/config.json admapfull.json
bash run_eval.sh
```

## Acknowledgements
Our code partially builds on [geomapnet](https://github.com/NVlabs/geomapnet).
