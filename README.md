# Project on Automatic Speaker Recognition
Course Project  on Speaker Identification using CNNs.

We use the [Voxceleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) dataset to train the CNN.

The implementation is based on the paper [VoxCeleb: a large-scale speaker identification dataset](http://www.robots.ox.ac.uk/~vgg/publications/2017/Nagrani17/nagrani17.pdf)

### Training the model : 

```
python train_model.py --save_path="SAVE" --log_path="Log"
```
where save_path is the save folder for the best model and log_path is the folder to store the automatic checkpoints.

**speaker_name.json** contains the mapping of speaker_id vs speaker name that can be used during testing. 



