# NeuroVision: Automated Neuron Counting Tool - AI4ALL '24 Project Repository

Developed an automated neuron counting tool using
a Residual Neural Network (ResNet) to analyze microscope images of brain tissue, all within AI4ALL's Ignite Accelerator. 

## Problem Statement <!--- do not change this line -->

Manual neuron counting is a time-consuming and error-prone
process that can impede neuroscience research. By automating this task, we
aim to enhance efficiency and accuracy, enabling researchers to process larger
datasets rapidly. This is essential for understanding brain development, studying neurological diseases, and evaluating treatment effectiveness, ultimately contributing to significant progress in neuroscience.

## Key Results <!--- do not change this line -->

1. *Applied a ResNet model to identify and count neuronal cells with reliable accuracy*
    - *Mean Error of -0.61 neurons*
    - *Mean Absolute Error of 1.94 neurons*
    - *Correlation Coefficient of 0.954*
    - *Pearson Correlation of 0.954*


## Methodologies <!--- do not change this line -->

To build the dataset, we created the NeuronDataset class, which loads pairs of microscope images and their corresponding segmentation masks from separate directories. The dataset applies standard transforms, like resizing, to ensure the inputs have a consistent size. Importantly, the dataset also counts the number of neurons present in each mask, providing the ground truth for the counting component. For the model architecture, we developed the NeuronCounter network, which has a ResNet-inspired design. The model has two main branches - one for segmentation and one for counting. The segmentation branch outputs a mask identifying the individual neurons, while the counting branch regresses the total neuron count. By combining these two tasks, the model can learn features that are useful for both segmentation and counting. During training, we implemented the train_model function, which handles the end-to-end training loop. It iterates through the dataset, computes a combined loss that includes both the segmentation and counting components, and then backpropagates the gradients to update the model parameters. To evaluate the model's performance, I wrote the evaluate_model function, which assesses the model on a held-out test set. This function computes the overall loss, as well as the individual segmentation and counting losses, giving me a comprehensive view of the model's capabilities. Finally, to enable model checkpointing and restoration, we created the save_checkpoint function, which saves the model, optimizer, and training state to a file.

## Data Sources <!--- do not change this line -->

Flourescent Neuronal Cells Dataset: [Link](https://amsacta.unibo.it/id/eprint/6706/)

## Technologies Used <!--- do not change this line -->

- Python
- torch
- numpy
- scipy


## Authors <!--- do not change this line -->


This project was completed in collaboration with:
- *Aaron Kim ([aarkim@tamu.edu](mailto:aarkim@tamu.edu))*
- *Ethan Carr ([ehc6@rice.edu](mailto:ehc6@rice.edu))*
- *Michael Tran ([michael.tran@tamu.edu](mailto:michael.tran@tamu.edu))*