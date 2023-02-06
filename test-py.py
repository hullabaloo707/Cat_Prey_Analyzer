import tensorflow as tf
import numpy as np

# Create a model using high-level tf.keras.* APIs
model  = tf.keras.models.load_model("models/Prey_Classifier/0.86_512_05_VGG16_ownData_FTfrom15_350_Epochs_2020_05_15_11_40_56.h5")
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(units=1, input_shape=[1]),
#     tf.keras.layers.Dense(units=16, activation='relu'),
#     tf.keras.layers.Dense(units=1)
# ])
# model.compile(optimizer='sgd', loss='mean_squared_error') # compile the model
# model.fit(x=[-1, 0, 1], y=[-3, -1, 1], epochs=5) # train the model
# # (to generate a SavedModel) tf.saved_model.save(model, "saved_model_keras_dir")

model.save("test.pb")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
# Post training quantization
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8


image_shape = (224,224,3)

# dataset_list = tf.data.Dataset.list_files("output" + '/*')
# print([*dataset_list])
#
# image = next(iter(dataset_list))
# print(image)
# image = tf.io.read_file(image)
# print(image)
#
# image = tf.io.decode_jpeg(image, channels=3)
# print("decode_jpeg")
# print(image)
#
# image = tf.image.resize(image, (224, 224))
# print("resize")
#
# print(image)
#
#
# image = tf.cast(image / 255., tf.float32)
# image = tf.expand_dims(image, 0)
# print(image)

def representative_data_gen():
    dataset_list = tf.data.Dataset.list_files("output" + '/*')
    for i in range(10):
        image = next(iter(dataset_list))
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image / 255., tf.float32)
        image = tf.expand_dims(image, 0)
        yield [image]

converter.representative_dataset = representative_data_gen


tflite_model = converter.convert()



# Save the model.
with open('PC.tflite', 'wb') as f:
    f.write(tflite_model)