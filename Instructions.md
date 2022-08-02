# Modulation spectrum course exercises:
This document will introduce dependencies needed to run the scripts as well as instructions for each exercise.<br />

This exercise is set up for people who are new to modulation spectrum to get some hands-on experience with using modulation spectrogram for speech (or other signals) analysis. After going through the exercise, participants should be able to use modulation spectrum for their own tasks. Two exercises are included: (1) ```ex1.py``` shows how to extract modulation spectrogram using the SRMR toolbox along with some visualizations; (2) ```ex2.py``` shows how to use the extracted modulation spectrogram features for disease detection using machine learning.

## Dependencies
-Programming language: Our current version only supports **Python** 3. <br />
-Packages: The SRMR toolbox is used to compute modulation spectrogram, which can be found at [https://github.com/jfsantos/SRMRpy].

## About exercises
To help participants better understand the use of modulation spectrogram, we designed two exercises. For simplicity and reproducibility, we include customized functions in the ```msr_func.py``` and ```msr_ml.py```. We will be using these customized functions for our exercises. For future usage, you can also modify these functions for your own tasks.
### Exercise 1: visualizing modulation spectrogram
In the first exercise, we are going to compute modulation spectrogram from a single speech recording, visualize it and compare with a conventional mel-spectrogram. Two toy recordings are included in the ```data/speech``` folder. One of them is collected from an individual with COVID-19, the other is from an individual without COVID-19. The speech samples included are from an open-source COVID-19 sound database **Coswara**.

#### Step 1: compute modulation spectrogram from each one of these recordings
To get started, we firstly load the audio file. The modulation spectrogram can then be computed using our customized function.

#### Step 2: visualize the temporally averaged modulation spectrogram
Since the extracted modulation spectrogram is 3-dimensional (3D), we average it over time into a 2D image for better visualization. <br />
-Try different parameters of modulation spectrogram, what difference do you notice? <br />
-Compare the modulation spectrogram with mel-spectrogram, what are the advantages and disadvantages of these two representations?

#### Step 3: self-explore
-Try with the other COVID/non-COVID recording, compare the plot with the previous one. Can you see any difference? <br />
-Try load your own audio file and make plots. Can you localize the discriminative region?


### Exercise 2: using modulation spectrogram for COVID-19 detection
In the second exercise, we will see how to use the extracted modulation spectrogram for COVID-19 detection. In this example, we will use the modulation spectrogram features that had been extracted beforehand. The features and labels are saved in the ```preprocessed``` folder as ```.pkl``` files.

#### Step 1: load saved files
We need to load the extracted features and labels to get started. Modulation spectrogram features are the flattened 2D modulation spectrogram that we saw in exercise 1, and labels are binary numeric numbers (0 for negative and 1 for positive).

#### Step 2: F-ratio plots
Before we directly input these features into a classifier, we can use the F-ratio to statistically quantify the difference between groups. The higher the F-ratio is, the more discrimination exists between two classes. Some useful info about F-ratio can be found at [https://sthalles.github.io/fisher-linear-discriminant/]. We customized the F-ratio caculation for unbalanced datasets, the script can be found at [https://github.com/zhu00121/Unbalanced_Fratio]. The F-ratio is computed iteratively for each feature, we then reshape these F-ratio values into 2D and visualize them in a topographical plot. Using this F-ratio plot, we can tell which modulation spectrogram region is more discriminative.

#### Step 3: data preparation
For the downstream machine learning model, we firstly need to split data into training and test sets. Then the feature values are normalized using a standard scaler (i.e., mean-std normlization). 

#### Step 4: model training
A linear-kernel support vector machine (SVM) model is chosen as the classifier for simplicity and reproducibility. For hyper-parameter tuning, we use a 3-fold cross-validation with the training set. You can refer to the customized functions in ```msr_ml.py``` for more details.

#### Step 5: model evaluation
Since this is an unbalanced dataset (with more non-COVID samples), we use AUC-ROC score rather than accuracy for model evaluation. The confusion matrix is also plotted to check how well this model performs for each class.
