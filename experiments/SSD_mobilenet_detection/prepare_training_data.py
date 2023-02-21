"""Naval Fate.

Usage:
  prepare_train_data.py [--number_pictures=<n> ][ --show_images][ --create_tf_records][ --detect][ --count_cats]

Options:
  -h --help     Show this screen.

"""
from docopt import docopt
from object_detection.utils import config_util

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import tqdm
import tensorflow as tf
from PIL import Image

from object_detection.utils import dataset_util
import numpy as np
import os
from object_detection.utils import visualization_utils as vis_util

import math

import cv2

category_index = {1: {'id': 1, 'name': 'cat'}}

record_file_train = 'data/images_train.tfrecords'
directory_train = "data/cat_data_set"
record_file_eval = 'data/images_eval.tfrecords'
directory_eval = "data/cat_data_set_eval"
config_file_path = os.path.join('models', "my_ssd_mobnet", 'pipeline.config')
my_checkpoints = os.path.join("models","my_ssd_mobnet")


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

# docker run -p 8080:8888 -v $(pwd):$(pwd)  -u $(id -u):$(id -g) -w $(pwd)/experiments/SSD_mobilenet_detection --gpus all -it --env NVIDIA_DISABLE_REQUIRE=1 test:gpu python models/research/object_detection/model_main_tf2.py --model_dir=models/my_ssd_mobnet --pipeline_config_path=models/my_ssd_mobnet/pipeline.config --num_train_steps=2000
# docker run -p 8080:8888 -v $(pwd):$(pwd)  -u $(id -u):$(id -g) -w $(pwd) --gpus all -it --env NVIDIA_DISABLE_REQUIRE=1 test:gpu bash -c "source venv/bin/activate && cd experiments/SSD_mobilenet_detection && python models/research/object_detection/model_main_tf2.py --model_dir=models/my_ssd_mobnet --pipeline_config_path=models/my_ssd_mobnet/pipeline.config --num_train_steps=2000"
@tf.function
def detect_fn(detection_model,image):
    # Load pipeline config and build a detection model
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def load_model_from_checkpoints():
    from object_detection.utils import config_util

    configs = config_util.get_configs_from_pipeline_file(config_file_path)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(my_checkpoints, 'ckpt-50')).expect_partial()
    return detection_model


def convert_and_filter_detections(detections,minimmum_detection_score):
    num_detections = int(detections.pop('num_detections'))
    # print(detections)
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    label_id_offset = 1
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64) +label_id_offset


    detections_filtered = {}
    for key, values in detections.items():
        if key == 'num_detections':
            continue
        detections_filtered[key] = np.array([v for (i,v) in enumerate(values) if detections["detection_scores"][i] > minimmum_detection_score])

    detections_filtered["num_detections"] = len(detections["detection_scores"])
    return detections_filtered

def is_class(model, image_path,class_id):

    img = cv2.imread(os.path.abspath(image_path))
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(model,input_tensor)

    minimmum_detection_score = 0.5
    detections_filtered = convert_and_filter_detections(detections,minimmum_detection_score)

    for i,x in enumerate(detections_filtered['detection_classes']):
        if x==class_id:
            return True


def show_detection(detection_model,image_path):
    from matplotlib import pyplot as plt

    img = cv2.imread(os.path.abspath(image_path))
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(detection_model,input_tensor)

    minimmum_detection_score = 0.5
    detections_filtered = convert_and_filter_detections(detections,minimmum_detection_score)
    # detection_classes should be ints.

    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections_filtered['detection_boxes'],
        detections_filtered['detection_classes'],
        detections_filtered['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=minimmum_detection_score,
        agnostic_mode=False)

    Image.fromarray(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)).show()
    plt.show()
    # plt.savefig("mygraph.png")


def main(arg):

    if arg["--detect"]:

        image_path = os.path.join("data","cat_data_set_eval","00000005_000.jpg")
        image_path = os.path.join("../../debug/input/","11-20201012212045-01.jpg")
        detection_model = load_model_from_checkpoints()
        show_detection(detection_model,image_path)
        return


    if arg["--count_cats"]:
        import pathlib

        PATH_TO_TEST_IMAGES_DIR = pathlib.Path('../../debug/output/input_cropped') #models/research/object_detection/test_images
        TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
        detection_model = load_model_from_checkpoints()
        total=0
        detected=0
        for image_path in TEST_IMAGE_PATHS:
            total+=1
            print(image_path)

            if is_class(detection_model, image_path, 1):
                detected+=1
                # show_detection(detection_model,image_path)
        print(f"total: {total}, detected: {detected}")
        return


        # detection_classes should be ints.

        label_id_offset = 1

        return

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
    # config = config_util.get_configs_from_pipeline_file(config_file_path)
    # print(config)
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(config_file_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    check_point_path = os.path.abspath(os.path.join("models","downloaded_models","datasets","ssd_mobilenet_v2_320x320_coco17_tpu-8","checkpoint",'ckpt-0'))
    pipeline_config.model.ssd.num_classes = len(category_index.items())
    pipeline_config.train_config.batch_size = 8 # good with multiple of 8
    pipeline_config.train_config.fine_tune_checkpoint = check_point_path
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path= label_map_file
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [record_file_train]
    pipeline_config.eval_input_reader[0].label_map_path = label_map_file
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [record_file_eval]
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [record_file_eval]
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = 1e-2
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_learning_rate = 1e-3
    pipeline_config.train_config.max_number_of_boxes = 100 # orginal 100
    pipeline_config.model.ssd.box_predictor.convolutional_box_predictor.use_dropout = True # orginal False
    # test with use_dropout
    # learning_rate_base 10e-3
    # warmup_learning_rate = 0
    # momentum_optimizer_value == friction
    config_text = text_format.MessageToString(pipeline_config)
    print(config_text)
    with tf.io.gfile.GFile(config_file_path, "wb") as f:
        f.write(config_text)

    TRAINING_SCRIPT = os.path.join("models", 'research', 'object_detection', 'model_main_tf2.py')

    command_clean_model = "rm -f models/my_ssd_mobnet/ckpt* && rm -f models/my_ssd_mobnet/checkpoint && rm -rf models/my_ssd_mobnet/train"

    command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=50000".format(TRAINING_SCRIPT, my_checkpoints,config_file_path)
    gpu_docker_command = f"docker run -p 8080:8888 -v $(pwd):$(pwd)  -u $(id -u):$(id -g) -w $(pwd) --gpus all -it --env NVIDIA_DISABLE_REQUIRE=1 test:gpu bash -c \"source venv/bin/activate && cd experiments/SSD_mobilenet_detection && {command_clean_model} && {command}\""

    print(gpu_docker_command)

    print("tensorboard --logdir=models/my_ssd_mobnet/train/")
    # command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, my_checkpoints,config_file_path, my_checkpoints)
    # print(gpu_docker_command)


# INFO:tensorflow:{'Loss/classification_loss': 2.2574234,
#                  'Loss/localization_loss': 0.5220827,
#                  'Loss/regularization_loss': 0.085143425,
#                  'Loss/total_loss': 2.8646493,
#                  'learning_rate': 0.000145}
#
# I0215 13:02:16.783209 140572399961920 model_lib_v2.py:705] Step 700 per-step time 0.859s
# INFO:tensorflow:{'Loss/classification_loss': 0.1964774,
# 'Loss/localization_loss': 0.28343034,
# 'Loss/regularization_loss': 0.08516403,
# 'Loss/total_loss': 0.56507176,
# 'learning_rate': 0.00041500002}
# I0215 13:22:38.758822 140572399961920 model_lib_v2.py:705] Step 2100 per-step time 0.871s
# INFO:tensorflow:{'Loss/classification_loss': 0.04290723,
# 'Loss/localization_loss': 0.0210552,
# 'Loss/regularization_loss': 0.085125566,
# 'Loss/total_loss': 0.149088,
# 'learning_rate': 0.0009999893}
# I0215 12:42:11.509295 140701104416576 model_lib_v2.py:705] Step 3900 per-step time 0.200s
# INFO:tensorflow:{'Loss/classification_loss': 0.02756812,
# 'Loss/localization_loss': 0.01577053,
# 'Loss/regularization_loss': 0.08503012,
# 'Loss/total_loss': 0.12836877,
# 'learning_rate': 0.000996139}


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