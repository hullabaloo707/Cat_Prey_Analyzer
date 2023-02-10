import tensorflow as tf
from PIL import Image

from object_detection.utils import dataset_util
import numpy as np
import os


def create_cat_tf_example(encoded_cat_image_data):
    """Creates a tf.Example proto from sample cat image.

   Args:
     encoded_cat_image_data: The jpg encoded data of the cat image.

   Returns:
     example: The created tf.Example.
   """

    height = 1032.0
    width = 1200.0
    filename = 'example_cat.jpg'
    image_format = b'jpg'

    xmins = [322.0 / 1200.0]
    xmaxs = [1062.0 / 1200.0]
    ymins = [174.0 / 1032.0]
    ymaxs = [761.0 / 1032.0]
    classes_text = ['Cat']
    classes = [1]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
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


def main():

    directory = "data/cat_data_set"
    for filename in os.listdir("data/cat_data_set"):
        file = os.path.join(directory, filename)
        print(file)

        if filename.endswith(".jpg"):
            image_np = np.array(Image.open(file))
            Image.fromarray(image_np).show()

        if filename.endswith("jpg.cat"):
            with open(file, "r") as features_file:
                annotations = features_file.readline().rstrip()
                print(annotations)

                annotations = annotations.split(" ")
                print(annotations)

                annotations = [int(i) for i in annotations]
                print(annotations)

                left_eye = (annotations[1],annotations[2])
                right_eye = (annotations[3],annotations[4])

    # writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

            return
    # TODO(user): Write code to read in your dataset to examples variable

    # for example in examples:
    #     tf_example = create_tf_example(example)
    #     writer.write(tf_example.SerializeToString())

    # writer.close()


if __name__ == '__main__':
    main()