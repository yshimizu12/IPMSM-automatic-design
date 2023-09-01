# IPMSM-automatic-design
Pytorch implementation of an automatic design system for interior permanent magnet synchronous motors (https://ieeexplore.ieee.org/document/9896140)

![generated_rotor_shapes](https://user-images.githubusercontent.com/75551755/143835458-ea355f78-fac2-4840-b9df-3b36253ba7ae.gif)

## Overview
This library contains a Pytorch implementation of an automatic design system for interior permanent magnet synchronous motors as presented in [[1]](https://ieeexplore.ieee.org/document/9896140)

## Dependencies
- python>=3.8
- torch>=1.6.0
- numpy
- pandas
- pillow

## Architecture
data: You can download the dataset used for the paper [here](https://ieee-dataport.org/documents/dataset-motor-parameters-ipmsm) or [here](https://www.kaggle.com/datasets/uuuuuuuuu/dataset-motor-parameters-ipmsm).  
[regression](/regression.py): A characteristics prediction model is implemented.  
GAN: See https://github.com/lucidrains/lightweight-gan

## Feedback
For questions and comments, feel free to contact [Yuki Shimizu](yshimizu@fc.ritsumei.ac.jp).

## License
MIT

## Citation
```
[1] Y. Shimizu, S. Morimoto, M. Sanada, and Y. Inoue, “Automatic Design System with Generative  
Adversarial Network and Convolutional Neural Network for Optimization Design of Interior  
Permanent Magnet Synchronous Motor,” IEEE Trans. Energy Convers., Vol. 38, No. 1, pp. 724-734, 2023.
```
