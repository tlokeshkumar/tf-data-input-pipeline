# tf-data-input-pipeline
  Ready made scripts to make data input and pre-processing easier in TensorFlow using Data API

Preprocessing data and configuring an input pipeline for a TensorFlow program can be a pain if enough experience is not present. In order to ease this process I have written a series of scripts to cover majority usecases so that you can focus more on the algorithm than on the data loading part.

Inspiration is taken from **Keras** and other sources to maximise support to variety of input pipelines.

```python
from input_utils import flow_from_directory

next_element, init_op = flow_from_directory('path/to/the/master/folder/of/dataset')

# Use next_element as input to the model

sess = tf.Session()
sess.run(init_op)

# Training Loop starts here
# Run the optimizer step multiple times


```

Support will be extended to Semantic Segmentation, text file parsing and others soon.
