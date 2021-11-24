# IPMSM-automatic-design
Pytorch implementation of an automatic design system of interior permanent magnet synchronous motor ([in preparation](https://www.techrxiv.org/))

## Overview
This library contrains a Pytorch implementation fo an automatic design system of interior permanent magnet synchronous motor as presented in '[1]'([in preparation](https://www.techrxiv.org/))

## Dependencies
- python>=3.8
- torch>=1.6.0
- numpy
- pandas
- pillow

## Usage
data: should contain your datasets. You can download the dataset used for the paper [here](https://ieee-dataport.org/documents/dataset-motor-parameters-ipmsm).  
regression: A characteristics prediction model is implemented. 
GAN: See https://github.com/lucidrains/lightweight-gan

## Feedback
For questions and comments, feel free to contact [Yuki Shimizu](de104004@edu.osakafu-u.ac.jp).

## License
MIT

## Citation
```
[1] Y. Shimizu, S. Morimoto, M. Sanada, and Y. Inoue, “Automatic Design System with Generative  
Adversarial Network and Convolutional Neural Network for Optimization Design of Interior  
Permanent Magnet Synchronous Motor,” in preparation
```

