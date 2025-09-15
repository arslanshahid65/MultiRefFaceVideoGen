# End to End Deep Learning Based Video Coding for Video Conferencing.
The project aims to develop a end to end video compression pipeline by using novel attention based multi reference fusion in existing First Order Motion Model to enhance video quality at ultra low bitrates. The attention model is an adaptation of non local neural network from Wang, Girshick, Gupta & He (2018) in _Non-local Neural Networks_ [arXiv:1711.07971](https://arxiv.org/abs/1711.07971).

## Model Implementation

The Multi-Frame Attention model is implemented in:
src/
└── models/
    └── inter-models/
        └── multi-frame-attention.py
 
