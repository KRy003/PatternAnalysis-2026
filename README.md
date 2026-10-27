# Pattern Analysis

I will attempt the easy target and increase the difficulty as i complete them, the line below will be the final version difficulty that i submit
Target Difficulty: Normal

Architecture: Improved U-Net

Title: Anatomical Contouring and Radiotherapy Planning.

Goal / Target Audience of Program: 
The goal of the program is to look at MRI images and automatically identify different structures of the body in the scans themselves, which in turns helps save radiotherapists time in needing to identify
the structures themselves and nearby organs which could be at risk, such as the bladder, rectum and femoral heads etc. This can also be used as a backup verification for the radiotherapists to make sure they're conclusions are similar to the AIs. 

The program is trained on data where it has already been identified for the program and it is then tested to verify the results on a test set. The algorithm itself follows a typical U-Net architecture
which is a type of neural network designed for image segmentation. For this particular problem there are considered 6 classes which are different unique structures found in the MRI and the goal is to distinguish between each of the classes.

Example of base MRI image with whats expected and the output of the code
![Results comparison](image-1.png)
Results from Training:
![Training And Validation loss / Validation Score from training](Loss_and_Dice_Validation-1.png)
![Individual Class Dice Scores](Class_Dice_Scores-1.png)

Basic U-Net Architecture:
Parameter 1 (Dice Scores):
Running the weights gathered from the training data and running the weights against all the images
i.e test images / validation images / training images
The score from each individual class was the following
===== Dice Scores =====
Dice 0: 0.9972
Dice 1: 0.9907
Dice 2: 0.9623
Dice 3: 0.9724
Dice 4: 0.9234
Dice 5: 0.9209

Average Dice Score: 0.9611
Average Dice Percentage: 96.11%

Parameter 2 (Run Time):
Running these results fresh gave a processing time of 80.74 milliseconds per image (running just once) however repeated runs showed that time per image actually dropped quite significantly. A second run of predict.py showed it goes to 20.43 milliseconds, and even further testing it can go down to roughly 18.21 (The lowest time after 10 runs). This would obviously change per system however it shows a rough average processing time around 20 millisecond per image.

Parameter 3 (Peak VRam):
Running the results gave a constant Peak VRAM of 674.42 MB of data per run of predict.py

Improved U-Net Training
The first improvement was trying to add residuals to the convolution layer. After 20 epoches of training this
one addition scored an average of 0.9303 across the test die
Epoch 20/20, Loss: 0.0252, Time: 64.87s
Test Dice:
  Class 0: 0.9971
  Class 1: 0.9846
  Class 2: 0.9354
  Class 3: 0.9627
  Class 4: 0.8623
  Class 5: 0.8537
Mean Dice: 0.9326
Upon Testing it it showed that the test set it was not as good either
===== Dice Scores =====
Dice 0: 0.9953
Dice 1: 0.9809
Dice 2: 0.9317
Dice 3: 0.8233
Dice 4: 0.8852
Dice 5: 0.8279

Average Dice Score: 0.9074
Average Dice Percentage: 90.74%
Peak VRAM: 738.77 MB
With increase VRAM and lower Dice Scores it's clear the residual setup currently implemented is not the best

5. AI Disclosure
The AI that has been used quite a bit in this is the visual studio code auto complete AI and it has been able to
auto complete alot of my Dataloading file and has helped with the module aswell although i did look up the
general architecture that a 2D U-net should follow. 
As for testing the AI code specifically it mostly comes from testing the algorithim all together

6. References
"UNet - Line by Line Explanation" Jeremy Zhang 2019
https://medium.com/data-science/unet-line-by-line-explanation-9b191c76baf5

