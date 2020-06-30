
# This is Multi MNIST As per Sara-Sabour's paper
import random
import torch
import numpy as np
from torchvision import datasets
from PIL import Image
from matplotlib import pyplot as plt

random.seed(4)
np.random.seed(4)
torch.manual_seed(4)
import torchvision.transforms as transforms



def shift_write_multi_mnist(input_dataset, file_prefix, shift, pad, max_shard,
                            num_pairs):
  """Writes the transformed duplicated data as tfrecords.
  Since the generated dataset is quite large, shards the output files. During
  writing selects the writer for each example randomly to diversify the range
  of labels in each file.
  Pads the data by adding zeros. Shifts all images randomly. For each image
  randomly selects a set of other images with different label as its pair.
  Aggregates the image pair with a maximum pixel value of 255.
  Writes overlayed pairs of shifted images as tf.train.Example in tfrecords
  files.
  Args:
    input_dataset: A list of tuples containing corresponding images and labels.
    file_prefix: String, prefix of the name of the resultant sharded tfrecord
      file.
    shift: Integer, the shift range for images.
    pad: Integer, the number of pixels to be padded.
    max_shard: Integer, maximum number of examples in each shard.
    num_pairs: Integer, number of pairs of images generated for each input
      image.
  """
  num_images = len(input_dataset)

  # writers, writer_turns = sharded_writers(num_images * num_pairs, max_shard,
  #                                         num_images, file_prefix)

  random_shifts = np.random.randint(-shift, shift + 1,
                                    (num_images, num_pairs + 1, 2))
  
  dataset = [(np.pad(image, pad, 'constant'), label)
             for (image, label) in input_dataset]

  for i, (base_image, base_label) in enumerate(dataset):
    
    # Shift each image
    base_shifted = shift_2d(base_image, random_shifts[i, 0, :], shift).astype(
        np.uint8)

    # Choose (2*num_pair) images out of num_images. 
    choices = np.random.choice(num_images, 2 * num_pairs, replace=False)
    chosen_dataset = []
    
    #
    for choice in choices:
      if dataset[choice][1] != base_label:
        chosen_dataset.append(dataset[choice])
    
    for j, (top_image, top_label) in enumerate(chosen_dataset[:num_pairs]):
      top_shifted = shift_2d(top_image, random_shifts[i, j + 1, :],
                             shift).astype(np.uint8)
      merged = np.add(base_shifted, top_shifted, dtype=np.int32)
      merged = np.minimum(merged, 255).astype(np.uint8)
      


      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'height': int64_feature(IMAGE_SIZE_PX + 2 * pad),
                  'width': int64_feature(IMAGE_SIZE_PX + 2 * pad),
                  'depth': int64_feature(1),
                  'label_1': int64_feature(base_label),
                  'label_2': int64_feature(top_label),
                  'image_raw_1': bytes_feature(base_shifted.tostring()),
                  'image_raw_2': bytes_feature(top_shifted.tostring()),
                  'merged_raw': bytes_feature(merged.tostring()),
              }))
      writers[writer_turns[i, j]].write(example.SerializeToString())

  for writer in writers:
    writer.close()