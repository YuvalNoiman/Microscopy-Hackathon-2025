# Moiré Fringes in High-Resolution Transmission Electron Microscopy   

# Hackathon Team
On The Fringe

# Team Members: 
_Hank Perry (University of Cincinnati)_
_Yuval Noiman (University of Cincinnati)_
_Jackie Cheng (University of Cincinnati)_

## Keywords
HRTEM, FFT, ML, Moiré Fringe, Crystal Structure

## Introduction
High-Resolution Transmission Electron Microscopy (HRTEM) is a fundamental tool for observing the atomic structures of materials. The samples used in HRTEM are so small that the electron beam passes through the entire thickness of the sample creating a 2D image of the 3D sample. When two crystalline lattices overlap it generates a complex interference pattern known as moiré fringe. The traditional way to analyze these patterns relies on manually doing geometrical phase analysis and then manually evaluating the moiré fringe which is time consuming and prone to losing the fine details [1], [2], [3], [4]. Methods that can automatically and accurately “de-stack” these layers into their constituent lattices are essential.

## Materials & Methodology
The primary objective of this project is to develop a machine-learning model capable of separating these overlapping crystalline lattices. We made some assumptions to make the model simpler. We assumed that there were only two overlapping layers, the atoms were gridded to only be in straight parallel lines, we can always orient one lattice to 0 degrees, the two layers have the same crystal structure, and that sample data is representative of real data.

## Dataset Generation
To start we created a moiré fringe generation script that would allow for the creation of synthetic data with different variations on rotation angle, atom size, lattice type, and vacancies. We created two datasets. The first had every atom the exact same size, no vacancies, and a square lattice type. The only thing being changed between the two layers was the angle of rotation. The second dataset we had randomly created vacancies, altered the rotation angle, chose between square and hexagonal lattice types, and allowed for two alternating atoms of different sizes.

## Model Preprocessing
The model uses Mean Squared Error, so before feeding the images into the model it uses a Poisson to convert the Poisson noise of a TEM image to a Gaussian variance.  The model uses a high pass filter to eliminate low frequency noise because the lattices it is looking for are high frequency.

## Model Encoding and Decoding
The model uses a dual encoder system.  One encoder works with the real space data received directly from the pre-processed TEM image, the other uses data after a Fast Fourier Transform (FFT) is applied to the same image and encodes the Fourier-space information of the image. The FFT uses the Fourier-space information and creates its own encoding based on the images, adding extra context to the model. The FFT is used to collect information on the crystal structure of the lattices. During encoding, both encoders downscale the image so the model can see bigger structure and not just focus on pixel-by-pixel changes.  Those embeddings are then combined into a single embedding that is then passed to a dual decoder system. Each decoder is built the exact same, unlike the encoders, but each is given a different goal image during training, this allows the model to focus each decoder on a specific part of the output.  In training the two images for the decoders would be the two single lattice images used to create the TEM image at the start.  The decoders progressively upscale the resolution of the image, reverting the downscaling down in the encoding phase.  The decoders then output an image each.  

## Loss Function
The loss function for the training is the sum of three Mean Squared Errors, one for each of the predicted single lattice images compared to the original single lattice image, and an error for the combination of the two predicted single lattices and the original double lattice picture.  The error for the combinations is multiplied by a variable that gradually rises linearly over the course of training to allow for less focus on reconstruction and more focus on the individual pictures early in the training process.  

## Results
When we ran the machine learning model on the two different datasets the result was the two substantially different layers. The model is able to somewhat-accurately predict what the lattice structures are based on the training data as seen in figures 2 and 3.

##	Discussion
In addition to the above model, models using pattern recognition and more standard convolutional neural network (CNN) for images were created, but neither provided great results most likely due to the lack of physics in the design of the models and since the problem is less like a standard computer vision problem, but more of a superposition problem.

## Conclusion
There is plenty of room to improve on this methodology and model in the future as well as additional testing. Running the model using an actual HRTEM moiré fringe dataset would provide better analysis of whether the model is accurate with real data. The more data used in training and the more epochs the model is, the run will most likely increase the accuracy of the predictions. 


## References

[1]	S. Kim et al., “Scanning moiré fringe imaging for quantitative strain mapping in semiconductor devices,” Appl. Phys. Lett., vol. 102, no. 16, p. 161604, Apr. 2013, doi: 10.1063/1.4803087.

[2]	A. Pofelski, S. Ghanad-Tavakoli, D. A. Thompson, and G. A. Botton, “Sampling optimization of Moiré geometrical phase analysis for strain characterization in scanning transmission electron microscopy,” Ultramicroscopy, vol. 209, p. 112858, Feb. 2020, doi: 10.1016/j.ultramic.2019.112858.

[3]	A. Pofelski et al., “2D strain mapping using scanning transmission electron microscopy Moiré interferometry and geometrical phase analysis,” Ultramicroscopy, vol. 187, pp. 1–12, Apr. 2018, doi: 10.1016/j.ultramic.2017.12.016.

[4]	X. Ke, M. Zhang, K. Zhao, and D. Su, “Moiré Fringe Method via Scanning Transmission Electron Microscopy,” Small Methods, vol. 6, no. 1, p. 2101040, 2022, doi: 10.1002/smtd.202101040.


