r"""Convert raw Microsoft COCO dataset to TFRecord for object_detection.
Attention Please!!!

1)For easy use of this script, Your coco dataset directory struture should like this :
    +Your coco dataset root
        +train2014
        +val2014
        +annotations
            -instances_train2014.json
            -instances_val2014.json
2)To use this script, you should download python coco tools from "http://mscoco.org/dataset/#download" and make it.
After make, copy the pycocotools directory to the directory of this "create_coco_tf_record.py"
or add the pycocotools path to  PYTHONPATH of ~/.bashrc file.

Example usage:
    python create_coco_tf_record.py --data_dir=/path/to/your/coco/root/directory \
        --set=train \
        --output_path=/where/you/want/to/save/pascal.record
        --shuffle_imgs=True
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pycocotools.coco import COCO
from PIL import Image
from random import shuffle
import os, sys
import numpy as np
import tensorflow as tf
import logging


SAMPLES_PER_FILES = 512

# from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir',
                    '/home/ace19/dl-data/COCO/download',
                    'Root directory to raw Microsoft COCO dataset.')
flags.DEFINE_string('set',
                    'train',    # 118287
                    'Convert training set or validation set')
# flags.DEFINE_string('output_dir',
#                     '/home/ace19/dl-data/COCO/tfrecord',
#                     'Path to output TFRecord')
flags.DEFINE_string('output_dir',
                    '/home/ace19/dl-data/COCO/fine-tuning_tfrecord',
                    'Path to reindexing output TFRecord')
flags.DEFINE_bool('shuffle_imgs',
                  True,
                  'whether to shuffle images of coco')
FLAGS = flags.FLAGS

# before fine tuning with coco 80 class, we must re-index of class
# old index:new index
fine_tuning_new_label = {
    1:15, 2:2, 3:7, 4:14, 5:1, 6:6, 7:19, 8:91, 9:4, 10:92,
    11:93, 13:94, 14:95, 15:96, 16:3, 17:8, 18:12, 19:13, 20:17, 21:10,
    22:22, 23:23, 24:24, 25:25, 27:27, 28:28, 31:31, 32:32, 33:33, 34:34,
    35:35, 36:36, 37:37, 38:38, 39:39, 40:40, 41:41, 42:42, 43:43, 44:5,
    46:46, 47:47, 48:48, 49:49, 50:50, 51:51, 52:52, 53:53, 54:54, 55:55,
    56:56, 57:57, 58:58, 59:59, 60:60, 61:61, 62:9, 63:18, 64:16, 65:65,
    67:11, 70:70, 72:20, 73:73, 74:74, 75:75, 76:76, 77:77, 78:78, 79:79,
    80:80, 81:81, 82:82, 84:84, 85:85, 86:86, 87:87, 88:88, 89:89, 90:90
}



def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_coco_dection_dataset(imgs_dir, annotations_filepath, shuffle_img = True ):
    """Load data from dataset by pycocotools. This tools can be download from "http://mscoco.org/dataset/#download"
    Args:
        imgs_dir: directories of coco images
        annotations_filepath: file path of coco annotations file
        shuffle_img: wheter to shuffle images order
    Return:
        coco_data: list of dictionary format information of each image
    """
    coco = COCO(annotations_filepath)
    img_ids = coco.getImgIds() # totally 82783 images
    cat_ids = coco.getCatIds() # totally 90 catagories, however, the number of categories is not continuous, \
                               # [0,12,26,29,30,45,66,68,69,71,83] are missing, this is the problem of coco dataset.
    # cats_text = coco.getCatsText()

    if shuffle_img:
        shuffle(img_ids)

    coco_data = []

    nb_imgs = len(img_ids)
    for index, img_id in enumerate(img_ids):
        if index % 100 == 0:
            print("Readling images: %d / %d "%(index, nb_imgs))
        img_info = {}
        bboxes = []
        labels = []
        # labels_text = []

        img_detail = coco.loadImgs(img_id)[0]
        pic_height = img_detail['height']
        pic_width = img_detail['width']

        ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            bboxes_data = ann['bbox']
            bboxes_data = [bboxes_data[0]/float(pic_width), bboxes_data[1]/float(pic_height),\
                                  bboxes_data[2]/float(pic_width), bboxes_data[3]/float(pic_height)]
                         # the format of coco bounding boxs is [Xmin, Ymin, width, height]
            bboxes.append(bboxes_data)
            # original coco
            # labels.append(ann['category_id'])
            # fine tuning with pre-trained checkpoint file so need to reindex
            labels.append(fine_tuning_new_label[ann['category_id']])
            # labels_text.append(cats_text.get(ann['category_id']).encode('utf-8'))


        img_path = os.path.join(imgs_dir, img_detail['file_name'])
        img_bytes = tf.gfile.FastGFile(img_path,'rb').read()

        img_info['pixel_data'] = img_bytes
        img_info['height'] = pic_height
        img_info['width'] = pic_width
        img_info['bboxes'] = bboxes
        img_info['labels'] = labels
        # img_info['labels_text'] = labels_text

        coco_data.append(img_info)
    return coco_data


def dict_to_coco_example(img_data):
    """Convert python dictionary formath data of one image to tf.Example proto.
    Args:
        img_data: infomation of one image, inclue bounding box, labels of bounding box,\
            height, width, encoded pixel data.
    Returns:
        example: The converted tf.Example
    """
    bboxes = img_data['bboxes']
    xmin, xmax, ymin, ymax = [], [], [], []
    for bbox in bboxes:
        xmin.append(bbox[0])
        xmax.append(bbox[0] + bbox[2])
        ymin.append(bbox[1])
        ymax.append(bbox[1] + bbox[3])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(img_data['height']),
        'image/width': _int64_feature(img_data['width']),
        'image/object/bbox/xmin': _float_feature(xmin),
        'image/object/bbox/xmax': _float_feature(xmax),
        'image/object/bbox/ymin': _float_feature(ymin),
        'image/object/bbox/ymax': _float_feature(ymax),
        'image/object/class/label': _int64_feature(img_data['labels']),
        # 'image/object/class/label_text': _bytes_feature(img_data['labels_text']),
        'image/encoded': _bytes_feature(img_data['pixel_data']),
        'image/format': _bytes_feature('jpeg'.encode('utf-8')),
    }))
    return example


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def main(_):
    if FLAGS.set == "train":
        imgs_dir = os.path.join(FLAGS.data_dir, 'train2017')
        annotations_filepath = os.path.join(FLAGS.data_dir,'annotations','instances_train2017.json')
        print("Convert coco train file to tf record")
    elif FLAGS.set == "val":
        imgs_dir = os.path.join(FLAGS.data_dir, 'val2017')
        annotations_filepath = os.path.join(FLAGS.data_dir,'annotations','instances_val2017.json')
        print("Convert coco val file to tf record")
    else:
        raise ValueError("you must either convert train data or val data")
    # load total coco data
    coco_data = load_coco_dection_dataset(imgs_dir, annotations_filepath, shuffle_img=FLAGS.shuffle_imgs)
    total_imgs = len(coco_data)

    n = 0
    idx = 0
    while n < total_imgs:
        # Open new TFRecord file.
        tf_filename = _get_output_filename(FLAGS.output_dir,
                                           FLAGS.set + '2017',
                                           idx)
        # write coco data to tf record
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            k = 0
            while n < total_imgs and k < SAMPLES_PER_FILES:
                if n % 100 == 0:
                    print("Converting images: %d / %d" % (n+1, total_imgs))
                example = dict_to_coco_example(coco_data[n])
                tfrecord_writer.write(example.SerializeToString())
                n += 1
                k += 1
            idx += 1


if __name__ == "__main__":
    tf.app.run()
