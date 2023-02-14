"""Naval Fate.

Usage:
  prepare_train_data.py [--number_pictures=<n> ][ --show_images][ --create_tf_records]

Options:
  -h --help     Show this screen.

"""
from docopt import docopt
from object_detection.utils import config_util

import tqdm
import tensorflow as tf
from PIL import Image

from object_detection.utils import dataset_util
import numpy as np
import os
from object_detection.utils import visualization_utils as vis_util

import math


category_index = {1: {'id': 1, 'name': 'cat'}}

record_file_train = 'data/images_train.tfrecords'
directory_train = "data/cat_data_set"
record_file_eval = 'data/images_eval.tfrecords'
directory_eval = "data/cat_data_set_eval"

def create_tf_example(file,image_height,image_width,file_name,bb_x_min,bb_y_min,bb_x_max,bb_y_max,class_text,class_value):
    """Creates a tf.Example proto from sample cat image.

   Args:
     encoded_cat_image_data: The jpg encoded data of the cat image.

   Returns:
     example: The created tf.Example.
   """

    with tf.io.gfile.GFile(file, 'rb') as fid:
        encoded_image_data = fid.read()

    # height = 1032.0
    # width = 1200.0
    # filename = 'example_cat.jpg'
    image_format = b'JPEG'
    colorspace = b'RGB'
    channels = 3

    xmins = [bb_x_min / image_width]
    xmaxs = [bb_x_max / image_width]
    ymins = [bb_y_min / image_height]
    ymaxs = [bb_y_max / image_height]
    classes_text = [bytes(class_text, encoding='utf-8')]
    classes = [class_value]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(image_height),
        'image/width': dataset_util.int64_feature(image_width),
        'image/colorspace': dataset_util.bytes_feature(colorspace),
        'image/channels': dataset_util.int64_feature(channels),
        'image/filename': dataset_util.bytes_feature(bytes(file_name, encoding='utf-8')),
        'image/source_id': dataset_util.bytes_feature(bytes(file_name, encoding='utf-8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_records(arg,directory,record_file):

    with tf.io.TFRecordWriter(record_file) as writer:
        for i, filename in enumerate(tqdm.tqdm(os.listdir(directory))):
            file = os.path.join(directory, filename)
            if filename.endswith(".jpg"):
                # print(file)
                image_np = np.array(Image.open(file))
                filename_annotation = filename + ".cat"
                file_annotation=os.path.join(directory, filename_annotation)
                if not os.path.exists(file_annotation):
                    print(f"skip: {filename}")
                    continue

                with open(os.path.join(directory, filename_annotation), "r") as features_file:
                    annotations = features_file.readline().rstrip()
                    annotations = annotations.split(" ")
                    annotations = [int(i) for i in annotations]

                    left_eye = np.array([annotations[2],annotations[1]])
                    right_eye = np.array([annotations[4],annotations[3]])
                    mouth = np.array([annotations[6],annotations[5]])
                    left_ear = np.array([annotations[10],annotations[9]])
                    right_ear = np.array([annotations[16],annotations[15]])

                    eye_vec = np.subtract(left_eye,right_eye)
                    distance_eyes = math.sqrt(math.pow(eye_vec[0],2)+math.pow(eye_vec[1],2))
                    # print(left_ear)
                    # print(left_eye)

                    image_y_max = image_np.shape[0]
                    image_x_max = image_np.shape[1]
                    y_min = int(min(left_eye[0],right_eye[0],mouth[0],left_ear[0],right_ear[0]) - 0.3*distance_eyes)
                    x_min = int(min(left_eye[1],right_eye[1],mouth[1],left_ear[1],right_ear[1])- 0.3*distance_eyes)
                    y_max = int(max(left_eye[0],right_eye[0],mouth[0],left_ear[0],right_ear[0])+ 0.8 * distance_eyes)
                    x_max = int(max(left_eye[1],right_eye[1],mouth[1],left_ear[1],right_ear[1])+ 0.3*distance_eyes)
                    y_min = max(y_min,0)
                    x_min = max(x_min,0)
                    y_max = min(y_max,image_y_max)
                    x_max = min(x_max,image_x_max)

                    boxes = np.array([np.hstack(((y_min,x_min),(y_max,x_max)))])

                    # print(boxes)
                    classes = np.array([1])
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        boxes,
                        classes,
                        None,
                        category_index)
                # writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
                if arg['--show_images']:
                    Image.fromarray(image_np).show()

                tf_example = create_tf_example(file,
                                               image_y_max,
                                               image_x_max,
                                               filename,
                                               x_min,
                                               y_min,
                                               x_max,
                                               y_max,
                                               "cat",
                                               1
                                               )
                writer.write(tf_example.SerializeToString())


                if arg['--number_pictures']:
                    if i > int(arg['--number_pictures']):
                        return
    writer.close()


def main(arg):

    if arg["--create_tf_records"]:
        create_tf_records(arg,directory_train,record_file_train)
        create_tf_records(arg,directory_eval,record_file_eval)

    label_map_file = os.path.join("models","my_ssd_mobnet","label_map.pbtxt")
    with open(label_map_file, 'w') as f:
        for key, value in category_index.items():
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(value['name']))
            f.write('\tid:{}\n'.format(value['id']))
            f.write('}\n')

    from object_detection.utils import config_util
    from object_detection.protos import pipeline_pb2
    from google.protobuf import text_format
    config_file_path = os.path.join('models', "my_ssd_mobnet", 'pipeline.config')
    # config = config_util.get_configs_from_pipeline_file(config_file_path)
    # print(config)
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_file_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    check_point_path = os.path.abspath(os.path.join("models","downloaded_models","datasets","ssd_mobilenet_v2_320x320_coco17_tpu-8","checkpoint",'ckpt-0'))
    pipeline_config.model.ssd.num_classes = len(category_index.items())
    pipeline_config.train_config.batch_size = 4
    pipeline_config.train_config.fine_tune_checkpoint = check_point_path
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path= label_map_file
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [record_file_train]
    pipeline_config.eval_input_reader[0].label_map_path = label_map_file
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [record_file_eval]

    config_text = text_format.MessageToString(pipeline_config)
    print(config_text)
    with tf.io.gfile.GFile(config_file_path, "wb") as f:
        f.write(config_text)

    TRAINING_SCRIPT = os.path.join("models", 'research', 'object_detection', 'model_main_tf2.py')

    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000".format(TRAINING_SCRIPT, os.path.join("models","my_ssd_mobnet"),config_file_path)
    print(command)


def check_jpg(directoy):
    from pathlib import Path
    import imghdr

    data_dir = directoy
    image_extensions = [".png", ".jpg"]  # add there all your images file extensions

    img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
    for filepath in Path(data_dir).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
                print(f"{filepath} is not an image")
            elif img_type not in img_type_accepted_by_tf:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")

if __name__ == '__main__':

    arguments = docopt(__doc__, version='0.0.0')
    print(arguments)
    check_jpg("data/cat_data_set_eval")
    check_jpg("data/cat_data_set")

    main(arguments)