# Pattern Analysis

I will attempt the easy target and increase the difficulty as i complete them, the line below will be the final version difficulty that i submit
Target Difficulty: Normal

Setup -------------------
The HIP MRI data should be in a file structure like
Target2.2_s47694413\Data\HIP MRI\keras_slices_data
This file should then contain all the keras_slices_seg_** and keras_slices_**
I can only 100% confidently say that latest version of train.py and predict.py are the most accurate versions of the code
however if needed to test earlier versions of the U-net you can possible download the modules.py file on the github commit and
run that with the train.py and predict.py, with a few modifications to train.py.
Uncomment lines 105 and 106 and comment lines 109-113 to test deep vision

Architecture: Improved U-Net

Title: Anatomical Contouring and Radiotherapy Planning.

Radiotherapy planning relies on identifying anatomical structures in medical images. Manually outlining these structures can be time-consuming, 
and inaccurate boundaries may affect treatment planning, especially near organs at risk. This project investigates whether a convolutional neural 
network can automatically segment anatomical regions in MRI images to assist radiotherapy professionals.

The intended users are radiotherapists and other clinicians involved in contouring and treatment planning. The model is a research prototype, 
not a clinically validated or autonomous diagnostic tool. Because it is still technically an AI its predictions would require an expert to review
any outputs it produces before any clinical use. However it could help achieve a cut down in time for potential experts in almost
immediately identifying key parts of the scan.

The dataset is treated as a six-class semantic segmentation problem: each image pixel is assigned one of six label IDs (0–5). 

------------------- Data Processing ---------------------------
The project uses HipMRI images and corresponding segmentation masks which are used for verification on training and testing. 
The model expects a single-channel image and predicts six class logits per pixel as mentioned before.
It should also be noted that alot of the images aren't the same dimension so alot were either forced to
scale up or down its vertical/horizontal axis to make sure that the input channels remain the same for
all cases.

------------------- Basic U-Net Architecture: -----------------
The baseline is an encoder–decoder convolutional network with three downsampling stages, a bottleneck, and three upsampling stages.
Each encoder stage applies two 3 × 3 convolutions with ReLU activations followed by 2 × 2 max pooling.
Decoder stages use transposed convolutions to upsample features, concatenate matching encoder features through 
skip connections, and apply two further convolutions.
A final 1 × 1 convolution generates six output logits per pixel.
The baseline feature widths are 32 → 64 → 128 → 256 at the bottleneck, then 128 → 64 → 32 through the decoder.
The predicted segmentation is obtained with argmax over the six output channels.
------------------- Training and Evaluation --------------------------
The models are trained using 2D MRI images and integer segmentation masks. 
The discussed training code uses CrossEntropyLoss, backpropagation and an optimizer to update model weights.
During training, the mean validation dice is used as a performance check and if the mean is higher
than previous epochs, then it saves the weights.
The learning rate is 0.001 and the training is run on 10 epochs.
------------------- Results ----------------------
Training basic U-Net:
Parameter 1 (Dice Scores):
Validation Dice:
  Class 0: 0.9953
  Class 1: 0.9823
  Class 2: 0.9328
  Class 3: 0.8810
  Class 4: 0.8778
  Class 5: 0.8208
Mean Validation Dice: 0.9150

Running the weights gathered from the training data and running the weights against all the test
images produced the following dice scores
The score from each individual class was the following

===== Dice Scores =====
Dice 0: 0.9972
Dice 1: 0.9846
Dice 2: 0.9351
Dice 3: 0.9631
Dice 4: 0.8365
Dice 5: 0.8449
Average inference time per image: 1.19 ms
Average Dice Score: 0.9269
Average Dice Percentage: 92.69%
Peak VRAM: 674.42 MB
Parameters ran on validation: 1925190

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
Validation Dice:
  Class 0: 0.9956
  Class 1: 0.9830
  Class 2: 0.9353
  Class 3: 0.9087
  Class 4: 0.8658
  Class 5: 0.8195
Mean Validation Dice: 0.9180
Upon Testing it it showed that the test set it was not as good either
Average inference time per image: 1.44 ms

===== Dice Scores =====
Dice 0: 0.9970
Dice 1: 0.9843
Dice 2: 0.9346
Dice 3: 0.9595
Dice 4: 0.8392
Dice 5: 0.8271

Average Dice Score: 0.9236
Average Dice Percentage: 92.36%
Peak VRAM: 738.76 MB
Parameters ran on validation: 2013106

With increase VRAM and lower Dice Scores it's clear the residual setup currently implemented is not the best

----------- Experiment Modification 1 and modification 2 together -----------------
With both modifications together it actually showed an improvement over the base U-Net architecture
    Validation Dice:
    Class 0: 0.9961
    Class 1: 0.9824
    Class 2: 0.9336
    Class 3: 0.8795
    Class 4: 0.8823
    Class 5: 0.8243
    Mean Validation Dice: 0.9164

===== Dice Scores =====
    Dice 0: 0.9971
    Dice 1: 0.9840
    Dice 2: 0.9336
    Dice 3: 0.9625
    Dice 4: 0.8393
    Dice 5: 0.8471
    Average inference time per image: 1.44 ms
    Average Dice Score: 0.9273
    Average Dice Percentage: 92.73%
    Peak VRAM: 738.76 MB
    Parameters ran on validation: 2013106

As seen from above the Average dice score on the test set is 92.7% compared to the 90.74% from the original U-Net
this means that its fails 26.85% less of the time.

----------- Modification 3 with Modification 2 and 1 all together
This is probably one of the more harder modifications to understand but this one essentially has
a sort of map to it called the attention gate which is another sort of learning feature that is
multiplied with a feature that can help spatially tell which part that the feature map to focus on.
Validation Dice:
  Class 0: 0.9953
  Class 1: 0.9825
  Class 2: 0.9356
  Class 3: 0.8554
  Class 4: 0.8808
  Class 5: 0.8315
Mean Validation Dice: 0.9135

===== Dice Scores =====
Dice 0: 0.9970
Dice 1: 0.9849
Dice 2: 0.9366
Dice 3: 0.9621
Dice 4: 0.8591
Dice 5: 0.8572
Average inference time per image: 1.60 ms
Average Dice Score: 0.9328
Average Dice Percentage: 93.28%
Peak VRAM: 802.86 MB
Parameters ran on validation: 2035397

3. Feasibility review — draft, expand to approximately one page
Technical feasibility. A 2D U-Net is a practical starting point for this dataset because its encoder captures context while skip connections retain spatial detail.
The recorded validation Dice scores of roughly around 0.91 indicate substantial overlap on the current validation split. 
Nevertheless, the split must be checked for patient-level independence, and results should be repeated under controlled settings to assess robustness.

Computational feasibility. The recorded peak allocated GPU memory is 802.86MB under the tested conditions for the final Improved U-Net architecture.
The increased memory usage of the attention model is measurable, while its mean validation Dice is slightly below the baseline.
Corrected inference benchmarks and parameter counts are required to judge deployment cost.

Clinical feasibility. Overlap metrics alone are insufficient to establish whether a segmentation is safe for radiotherapy planning.
Errors near small structures and organ boundaries may be more important than errors in large, easy regions.
This project has not demonstrated clinical safety, external generalization, or clinician acceptance.
Any use would require professional verification and further validation on independent patients and institutions.

Project feasibility. The modular PyTorch implementation makes it possible to compare architectural changes using common loaders and evaluation functions.
The main remaining tasks are to confirm data provenance and splitting, rerun a clean held-out test comparison, collect corrected resource measurements, and analyze representative failures.

4. Engineering recommendation — provisional
Current recommendation: retain the standard 2D U-Net as the preferred baseline until the improved architectures demonstrate a reproducible advantage on clinically relevant structures or boundaries. In the recorded validation comparison, the baseline achieved a slightly higher mean Dice (0.9148) while using less GPU memory than the combined modified models. Attention gates produced small improvements in some classes but increased measured peak memory.

This recommendation is provisional. A final decision should use held-out patient-level test results, per-class Dice/IoU, boundary error inspection, corrected inference latency, memory and parameter count. If the improved model consistently reduces important boundary errors, the additional computational cost may be justified even without a higher mean Dice.

5. AI Disclosure
The AI that has been used quite a bit in this is the visual studio code auto complete AI and it has been able to
auto complete alot of my Dataloading file and has helped with the module aswell although i did look up the
general architecture that a 2D U-net should follow. 
As for testing the AI code specifically it mostly comes from testing the algorithim all together
Chatgpt has been used primarily for the ideas of the improved system and the correct implementation.


6. References
"UNet - Line by Line Explanation" Jeremy Zhang 2019
https://medium.com/data-science/unet-line-by-line-explanation-9b191c76baf5

