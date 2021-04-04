# PSMB-Net
A whole-body PET/CT tumor segmentation method using position supervised multi-branch neural network.

## 1 Introduction
This project is a position supervised multi-branch neural network (PSMB-Net), which is a segmentation method for whole-body PET/CT images. This method is CNN-based, utilizing relative position of slices, and a multi-branch structural design for feature extraction and fusion. We separate whole-body to two semi-bodies by the lower bound of lung to reduce learning difficulty. To accommodate the remarkable visual differences of two semi-bodies, our design includes two decoders to learn features from them separately. The relative position of slices is used to guide the separation of PET/CT images into upper and lower semi-bodies. It also embeds inherent location information in feature maps to guide segmentation.

## 2 Method
The PSMB-Net consists of four components, including (A) positional learning on the CT slices, (B) CT feature fusion, (C) gate signal generation, and (D) PET/CT feature fusion and gating.

<img width="1031" alt="image" src="https://user-images.githubusercontent.com/71493468/113496441-29f5b400-952c-11eb-9ff5-30eb021f64da.png">
<img width="646" alt="image" src="https://user-images.githubusercontent.com/71493468/113496751-4b0bd400-952f-11eb-94db-84e7ac796d0e.png">

## 3 Experiments
Our data included de-identified whole-body PET/CT scans from 480 lung cancer patients. Dice, Sensitivity, and Precision are as metrics to evaluate the performance. Our PSMB-Net archived 0.580, 0.616, and 0.688, for Dice, Sensitivity and Precision, respectively.

## 4 Code usage
This code is based on python 3.7.3 and Tensorflow 1.14.0.

Using this code by running main.py, several settings are provided in the top of this file to fit a specific environment. Following is our recommanded using process:

(1) Manual assign two slices as reference for each patient,one slice is at the level of greater tubercle of the humerus and the other slice is at the level of the lumbar vertebrae.

(2) Calculate tumor-wise balance factor for each tumor. The voxel value of tumor label can be used to storing the factor, and an element-wise multiply of tumor label and probability map after softmax is performed during loss calculation.

(3) Prepare data. Using a .tfrecord file to store all the modalities and labels of a patient. The image should be volumetric, the shape should be (slice row pixel number, slice column pixel number, slice number). A random shuffled list, with each element of the list is the patient name, should be stored in disk. This can be implemented by using python package "pickle".

(4) Change the settings in main.py to fit your environments, and set the mode='train' to start training. If the training process is interupted by incident, you can set restore='True' and mode='train' to restore the parameters before the incident.

(5) In test stage, set mode='test' and running. For each patient, a volumetric image will be written to disk, which composed by four parts: Left upper part is PET original image, left lower part is tumor label, right upper part is tumor probability map, and right lower part is fully dark to generate tumor prediction by using CRF.

(6) Running multi_process_crf.py to use CRF to generate hard tumor label. This will reading the stored volumetric image in step 5 and write predicted tumor in right lower part. After all the patients completed, it will calculate Dice, Sensitivity and Precision for each patient and the average value of three metrics, the result will be written to a .xlsx file.
