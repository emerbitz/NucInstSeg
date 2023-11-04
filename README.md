# Nuclei instance segmentation - NucInstSeg
Nuclei instance segmentation is a crucial task in biomedical image analysis. However, nuclei instance segmentation is also a challenging task due to densely clustered nuclei and heterogeneity in the nuclei phenotype including size and shape. Numerous postprocessing methods suitable for deep-learning-based nuclei instance segmentation have been proposed. This project aims to compare the following postprocessing methods:
-	A contour-based postprocessing method that is similar to the postprocessing methods of Chen *et al*. 2017 and Zhou *et al*. 2019
-	The postprocessing method from “Nuclei segmentation using marker-controlled watershed, tracking using mean-shift, and Kalman filter in time-lapse microscopy” by Yang *et al*. 2006
-	The postprocessing method from “Segmentation of nuclei in histopathology images by deep regression of the distance map” by Naylor *et al*. 2019
-	The postprocessing method from “HoVer-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images” by Graham *et al*. 2019
-	A modified version of the postprocessing method from Graham *et al*. 2019


The postprocessing methods are further compared with a baseline postprocessing method incapable of separating touching/overlapping nuclei to allow for a better performance evaluation of the postprocesing methods. 

The evaluation of the postprocessing methods is conducted using 
-	the ground truth representations as input to determine the upper performance limit of the postprocessing methods.
-	a single and a dual decoder U-Net (Ronneberger *et al*. 2015) to establish a baseline performance.
-	a REU-Net (Qin *et al*. 2022) as a state-of-the-art network.
