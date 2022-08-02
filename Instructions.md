# Modulation spectrum course exercises:
This document will introduce dependencies needed to run the scripts as well as instructions for each exercise.<br />

This exercise is set up for people who are new to modulation spectrum to get some hands-on experience with using modulation spectrogram for speech (or other signals) analysis. After going through the exercise, participants should be able to use modulation spectrum for their own tasks. Two exercises are included: (1) **ex1.py** shows how to extract modulation spectrogram using the SRMR toolbox along with some visualizations; (2) **ex2.py** shows how to use the extracted modulation spectrogram features for disease detection using machine learning.

## Dependencies
Programming language: Our current version only supports **Python** 3. <br />
Packages: The SRMR toolbox is used to compute modulation spectrogram, which can be found at .

## About exercises
To help participants better understand the use of modulation spectrogram, we designed two exercises. For simplicity and reproducibility, we include customized functions in the [msr_func.py] and [msr_ml.py]. We will be using these customized functions for our exercises. For future usage, you can also modify these functions for your own tasks.
### Exercise 1: visualizing modulation spectrogram
In the first exercise, we are going to compute modulation spectrogram from a single speech recording, visualize it and compare with a conventional mel-spectrogram. Two toy recordings are included in the *data/speech* folder. One of them is collected from an individual with COVID-19, the other is from an individual without COVID-19. The speech samples included are from an open-source COVID-19 sound database *Coswara*.

#### Step 1: compute modulation spectrogram from each one of these recordings
To get started, we firstly load the audio file. The modulation spectrogram can then be computed using our customized function.

#### Step 2: visualize the temporally averaged modulation spectrogram
Since the extracted modulation spectrogram is 3-dimensional, we average it over time into a 2D image for better visualization. <br />
Try different parameters of modulation spectrogram, what difference do you notice? <br />
Compare the modulation spectrogram with mel-spectrogram, what are the advantages and disadvantages of these two representations?

#### Step 3: self-explore
Try with the other COVID/non-COVID recording, compare the plot with the previous one. Can you see any difference? <br />
Try load your own audio file and make plots. Can you localize the discriminative region?


### Exercise 2: using modulation spectrogram for COVID-19 detection
