# Pattern Analysis
Pattern Analysis of various datasets by COMP3710 students in 2026 at the University of Queensland.

We create pattern recognition and image processing library for Tensorflow (TF), PyTorch or JAX.

This library is created and maintained by The University of Queensland [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) students.

The library includes the following implemented in Tensorflow:
* fractals 
* recognition problems

In the recognition folder, you will find many recognition problems solved including:
* segmentation
* classification
* graph neural networks
* StyleGAN
* Stable diffusion
* transformers
etc.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
I will attempt the easy target and increase the difficulty as i complete them, the line below will be the final version difficulty that i submit
Target Difficulty: NO TARGET ATM

Architecture: Improved U-Net

Easy Implementation:
1. Project Overview:
The easy objective is to implement and evaluate a Improved U-Net architecture for the brain MRI scans found in the OASIS
dataset. The brains are to be split up into 3 seperate categories of matter, Delineating cortical gray matter, white matter, and
cerebrospinal fluid, and the U-Net must achieve a 90% dice accuracy for each category of matter, (class 0 is anything none brain matter).

2. Model Architecture:
For the easy task the current U-Net architecture looks like the following
    - Double Convolution: 2x (2dConvultion's followed by a RELU)
    - Down Sampling: A Double Convultion followed by a 2dmaxpool with stride of 2
    - Up Sampling: A 2d Convultion Transpose followed by a Double Convultion. The output is also concatenated with the Down Sampled data.
    - U-NET: 3x (Down Sampling) Followed by a bottleneck (Double Convultion) Followed by 3x (Up Sampling) Followed by a 2d Convultion

3. Training Configuration:
    - Framework PyTorch
    - Optimiser: Adam
    - Batch Size: 16
    - Learning Rate: 0.001
    - Loss Function: Corss-Entropy

4. Results:
The output of the easy implemenetation after 10 epochs is the following 
Epoch 10/10, Loss: 0.0330, Time: 57.29s
Validation Dice:
  Class 0: 0.9966
  Class 1: 0.9236
  Class 2: 0.9539
  Class 3: 0.9746
Mean Dice: 0.9622
Test Dice:
  Class 0: 0.9955
  Class 1: 0.9205
  Class 2: 0.9502
  Class 3: 0.9732
Mean Dice: 0.9598

5. References
"UNet - Line by Line Explanation" Jeremy Zhang 2019
https://medium.com/data-science/unet-line-by-line-explanation-9b191c76baf5