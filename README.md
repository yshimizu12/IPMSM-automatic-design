# IPMSM-automatic-design
Pytorch implementation of an automatic design system for interior permanent magnet synchronous motors ([in preparation](https://www.techrxiv.org/))
![generated_rotor_shapes](https://user-images.githubusercontent.com/75551755/143834674-f3168a9a-dd54-4ced-aba9-c01aaf5fee41.gif)

## Overview
This library contains a Pytorch implementation of an automatic design system for interior permanent magnet synchronous motors as presented in [1] ([in preparation](https://www.techrxiv.org/))

## Dependencies
- python>=3.8
- torch>=1.6.0
- numpy
- pandas
- pillow

## Usage
data: should contain your datasets. You can download the dataset used for the paper [here](https://ieee-dataport.org/documents/dataset-motor-parameters-ipmsm).  
[regression](/regression.py): A characteristics prediction model is implemented.  
GAN: See https://github.com/lucidrains/lightweight-gan

## Feedback
For questions and comments, feel free to contact [Yuki Shimizu](de104004@edu.osakafu-u.ac.jp).

## License
MIT

## Citation
```
[1] Y. Shimizu, S. Morimoto, M. Sanada, and Y. Inoue, “Automatic Design System with Generative  
Adversarial Network and Convolutional Neural Network for Optimization Design of Interior  
Permanent Magnet Synchronous Motor,” Submitted
```
