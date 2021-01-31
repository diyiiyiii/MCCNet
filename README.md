# Arbitrary Video Style Transfer via Multi-Channel Correlation --It is the re-implement of MCCNet on Jittor
Yingying Deng, Fan Tang, Weiming Dong, Haibin Huang, Chongyang Ma, Changsheng Xu  <br>

## Experiment
### Requirements
* python 3.6
* pytorch 1.4.0
* jittor 1.2.2.24
* PIL, numpy, scipy
* tqdm  <br> 


### Testing 
Pretrained models: [vgg-model](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing),  [decoder],  [MCC_module](see above)   <br> 
Please download them and put them into the floder  ./experiments/  <br> 
```
python JTtest_video.py  --content_dir input/content/ --style_dir input/style/    --output out
```
### Training  
Traing set is WikiArt collected from [WIKIART](https://www.kaggle.com/c/painter-by-numbers )  <br>  
Testing set is COCO2014  <br>  
```
python JTtrain.py --style_dir ../../datasets/Images --content_dir ../../datasets/train2014 --save_dir models/ --batch_size 4
```
### Reference
If you use our work in your research, please cite us using the following BibTeX entry ~ Thank you ^ . ^. Paper Link [pdf](coming soon)<br> 
```
@inproceedings{deng:2020:arbitrary,
  title={Arbitrary Video Style Transfer via Multi-Channel Correlation},
  author={Deng, Yingying and Tang, Fan and Dong, Weiming and Huang, haibin and Ma chongyang and Xu, Changsheng},
  booktitle={AAAI},
  year={2021},
 
}
```
