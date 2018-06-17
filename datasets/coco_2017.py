# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the COCO Dataset (images + annotations).
"""
import os

import tensorflow as tf

from datasets import dataset_utils


slim = tf.contrib.slim

FILE_PATTERN = '%s2017_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.'
}
# # (Images, Objects) statistics on every class.
# TRAIN_STATISTICS = {
#     'none': (0, 0),
#     'aeroplane': (670, 865),
#     'bicycle': (552, 711),
#     'bird': (765, 1119),
#     'boat': (508, 850),
#     'bottle': (706, 1259),
#     'bus': (421, 593),
#     'car': (1161, 2017),
#     'cat': (1080, 1217),
#     'chair': (1119, 2354),
#     'cow': (303, 588),
#     'diningtable': (538, 609),
#     'dog': (1286, 1515),
#     'horse': (482, 710),
#     'motorbike': (526, 713),
#     'person': (4087, 8566),
#     'pottedplant': (527, 973),
#     'sheep': (325, 813),
#     'sofa': (507, 566),
#     'train': (544, 628),
#     'tvmonitor': (575, 784),
#     'total': (11540, 27450),
# }
SPLITS_TO_SIZES = {
    'train': 118287,
    'val': 5000
}
# SPLITS_TO_STATISTICS = {
#     'train': TRAIN_STATISTICS,
# }
NUM_CLASSES = 80


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if not file_pattern:
        file_pattern = os.path.join(dataset_dir, FILE_PATTERN % split_name)
    else:
        file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], dtype=tf.int64),
        'image/width': tf.FixedLenFeature([1], dtype=tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
        # 'image/object/class/label_text': tf.FixedLenFeature((), tf.string, default_value=''),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
        # 'object/label_text': slim.tfexample_decoder.Tensor('image/object/class/label_text'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)
    # else:
    #     labels_to_names = create_readable_names_for_imagenet_labels()
    #     dataset_utils.write_label_file(labels_to_names, dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
        num_classes=NUM_CLASSES,
        labels_to_names=labels_to_names
    )

