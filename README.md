# CNN
To run the CNN models, you must first have the corresponding PyTorch libraries installed. A GPU is preferred as it will make the training significantly faster.

To train the models, run train.py with the specified arguments: 
1. CNN
2. CNN_augmented
3. CNN_mixup
4. CNN_cutout
5. CNN_both
6. ResNet
7. ResNet_augmented
8. ResNet_mixup
9. ResNet_cutout
10. ResNet_both

Eg. Running 
```buildoutcfg
train.py ResNet
```
 will train and report validation of a ResNet model with no data augmentation. The best checkpoint and final checkpoints of the models are saved in the checkpoints folder. Validation results are saved as a pickle file in the valiation_results folder. 
 
 The models can be tested using test.py by editing the source of the checkpoint to be loaded. 
 
 data.py and plots.py are used to visualize data and graph plots.