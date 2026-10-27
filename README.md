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
Validation Dice:
  Class 0: 0.9954
  Class 1: 0.9828
  Class 2: 0.9333
  Class 3: 0.8948
  Class 4: 0.8781
  Class 5: 0.8046
Mean Validation Dice: 0.9148

Running the weights gathered from the training data and running the weights against all the images
i.e test images / validation images / training images
The score from each individual class was the following

===== Dice Scores =====
Dice 0: 0.9971
Dice 1: 0.9846
Dice 2: 0.9340
Dice 3: 0.9653
Dice 4: 0.8445
Dice 5: 0.8502

Average Dice Score: 0.9293
Average Dice Percentage: 92.93%

Parameter 2 (Run Time):
Running these results fresh gave a processing time of 80.74 milliseconds per image (running just once) however repeated runs showed that time per image actually dropped quite significantly. A second run of predict.py showed it goes to 20.43 milliseconds, and even further testing it can go down to roughly 18.21 (The lowest time after 10 runs). This would obviously change per system however it shows a rough average processing time around 20 millisecond per image.

Parameter 3 (Peak VRam):
Running the results gave a constant Peak VRAM of 674.42 MB of data per run of predict.py

Improved U-Net Training
-------------------- First Modification -------------------------
NOTE: Due to some mistakes the individual scores of the First and Second Modifications on their own are wrong, However the scores showing both of them
combined and onwards are correct.

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
--------------------- Second Modification ------------------------------
The second modification was trying to store a loss at each convolution stage and additing it to the
loss function at the end with a specific magnitude, so the loss for the first convultion was added
with a factor of 0.2, the second convultion with a factor of 0.3 and the last convolution with a factor
of 1. These were the results of the following
Test Dice:
  Class 0: 0.9971
  Class 1: 0.9847
  Class 2: 0.9361
  Class 3: 0.9652
  Class 4: 0.8486
  Class 5: 0.8278
Mean Dice: 0.9266

===== Dice Scores =====
Dice 0: 0.9952
Dice 1: 0.9819
Dice 2: 0.9336
Dice 3: 0.8617
Dice 4: 0.8753
Dice 5: 0.8256

Average Dice Score: 0.9122
Average Dice Percentage: 91.22%
Peak VRAM: 674.43 MB

----------- Experiment Modification 1 and modification 2 together -----------------
With both modifications together it actually showed an improvement over the base U-Net architecture
Validation Dice:
  Class 0: 0.9957
  Class 1: 0.9826
  Class 2: 0.9361
  Class 3: 0.8659
  Class 4: 0.8784
  Class 5: 0.8219
Mean Validation Dice: 0.9134

===== Dice Scores =====
Dice 0: 0.9973
Dice 1: 0.9845
Dice 2: 0.9355
Dice 3: 0.9627
Dice 4: 0.8301
Dice 5: 0.8517

Average Dice Score: 0.9270
Average Dice Percentage: 92.70%
Peak VRAM: 738.76 MB

As seen from above the Average dice score on the test set is 92.7% compared to the 90.74% from the original U-Net
this means that its fails 26.85% less of the time.

----------- Modification 3 with Modification 2 and 1 all together
This is probably one of the more harder modifications to understand but this one essentially has
a sort of map to it called the attention gate which is another sort of learning feature that is
multiplied with a feature that can help spatially tell which part that the feature map to focus on.
Validation Dice:
  Class 0: 0.9955
  Class 1: 0.9826
  Class 2: 0.9356
  Class 3: 0.8554
  Class 4: 0.8860
  Class 5: 0.8248
Mean Validation Dice: 0.9133

===== Dice Scores =====
Dice 0: 0.9972
Dice 1: 0.9849
Dice 2: 0.9366
Dice 3: 0.9638
Dice 4: 0.8413
Dice 5: 0.8447

Average Dice Score: 0.9281
Average Dice Percentage: 92.81%
Peak VRAM: 802.86 MB


5. AI Disclosure
The AI that has been used quite a bit in this is the visual studio code auto complete AI and it has been able to
auto complete alot of my Dataloading file and has helped with the module aswell although i did look up the
general architecture that a 2D U-net should follow. 
As for testing the AI code specifically it mostly comes from testing the algorithim all together

6. References
"UNet - Line by Line Explanation" Jeremy Zhang 2019
https://medium.com/data-science/unet-line-by-line-explanation-9b191c76baf5

