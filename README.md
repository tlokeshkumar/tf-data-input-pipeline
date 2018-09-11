# tf-data-input-pipeline
  Ready made scripts to make data input and pre-processing easier in TensorFlow using Data API

Preprocessing data and configuring an input pipeline for a TensorFlow program can be a pain if enough experience is not present. In order to ease this process I have written a series of scripts to cover majority usecases so that you can focus more on the algorithm than on the data loading part.

Inspiration is taken from **Keras** and other sources to maximise support to variety of input pipelines.

Use `.jpg` images. The extension to other formats will be provided.


## Image Classification
```python
from input_utils import flow_from_directory

next_element, init_op = flow_from_directory('path/to/the/master/folder/of/dataset')

# Use next_element as input to the model

sess = tf.Session()
sess.run(init_op)

# Training Loop starts here
# Run the optimizer step multiple times


```

## Semantic Segmentation
```python
from input_utils import segmentation_data

next_element, init_op = segmentation_data(image_path, mask_path)

# Use next_element as input to the model

sess = tf.Session()
sess.run(init_op)

image, mask = sess.run(next_element)
# Training Loop starts here
# Run the optimizer step multiple times

```

## Single Image Input

Used in cases of
  - Unsupervised images training
  - **Super Resolution** - Where low resolution image is obtained from the high resolution images. So effectively from one image the other pair can be found. (other tasks include **deblurring, denoising** etc)
  
Provision has been provided to code the function that maps one image to another. The code by default implements a scaling (resizing by a factor of 2). Support can be extended based on user needs.

You can also work with the same image (in case of auto encoders) where you can remove the line where the function is written. (Integration with the function arguments will be done)

```python
from input_utils import read_no_labels

# image_path contains the path to the folder of images
image_path = '/home/tlokeshkumar/Downloads/GTOS_256/h_sample002_01'
next_element, init_op = read_no_labels(image_path)

# Use next_element as input to the model

sess = tf.Session()
sess.run(init_op)

# Here image represents the image in the folder given above. mask corresponds to the transformed version of the input image. (in case of super resolution the low res image if inputs are high res image)

image, mask = sess.run(next_element)

# Training Loop starts here
# Run the optimizer step multiple times

```

Support will be extended to other methods of input also.
