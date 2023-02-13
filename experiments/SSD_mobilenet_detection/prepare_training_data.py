import tensorflow as tf
from PIL import Image

from object_detection.utils import dataset_util
import numpy as np
import os
from object_detection.utils import visualization_utils as vis_util

import math

def create_tf_example(encoded_image_data,image_height,image_width,file_name,bb_x_min,bb_y_min,bb_x_max,bb_y_max,class_text,class_value):
    """Creates a tf.Example proto from sample cat image.

   Args:
     encoded_cat_image_data: The jpg encoded data of the cat image.

   Returns:
     example: The created tf.Example.
   """

    # height = 1032.0
    # width = 1200.0
    # filename = 'example_cat.jpg'
    image_format = b'jpg'

    xmins = [bb_x_min / image_width]
    xmaxs = [bb_x_max / image_width]
    ymins = [bb_y_min / image_height]
    ymaxs = [bb_y_max / image_height]
    classes_text = [bytes(class_text, encoding='utf-8')]
    classes = [class_value]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(image_height),
        'image/width': dataset_util.int64_feature(image_width),
        'image/filename': dataset_util.bytes_feature(bytes(file_name, encoding='utf-8')),
        'image/source_id': dataset_util.bytes_feature(bytes(file_name, encoding='utf-8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data.tobytes()),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main():

    record_file = 'images.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:


        directory = "data/cat_data_set"
        counter = 0
        for i, filename in enumerate(os.listdir("data/cat_data_set")):
            file = os.path.join(directory, filename)
            print(file)

            if filename.endswith(".jpg"):
                image_np = np.array(Image.open(file))
                filename_annotation = filename + ".cat"
                print(filename_annotation)

                with open(os.path.join(directory, filename_annotation), "r") as features_file:
                    annotations = features_file.readline().rstrip()
                    print(annotations)

                    annotations = annotations.split(" ")
                    print(annotations)

                    annotations = [int(i) for i in annotations]
                    print(annotations)

                    left_eye = np.array([annotations[2],annotations[1]])
                    right_eye = np.array([annotations[4],annotations[3]])
                    mouth = np.array([annotations[6],annotations[5]])
                    left_ear = np.array([annotations[10],annotations[9]])
                    right_ear = np.array([annotations[16],annotations[15]])

                    eye_vec = np.subtract(left_eye,right_eye)
                    distance_eyes = math.sqrt(math.pow(eye_vec[0],2)+math.pow(eye_vec[1],2))
                    boxes = np.array([np.hstack((left_eye,right_eye))])
                    print(left_ear)
                    print(left_eye)

                    boxes = np.array([np.hstack((left_eye,right_eye))])

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


                    print(boxes)
                    classes = np.array([1])
                    category_index = {1: {'id': 1, 'name': 'cat'}}
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        boxes,
                        classes,
                        None,
                        category_index)
        # writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

                Image.fromarray(image_np).show()
                tf_example = create_tf_example(image_np,
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



                if i > 13:
                    return
    # TODO(user): Write code to read in your dataset to examples variable

    # for example in examples:
    #     tf_example = create_tf_example(example)
    #     writer.write(tf_example.SerializeToString())

    # writer.close()


if __name__ == '__main__':
    main()