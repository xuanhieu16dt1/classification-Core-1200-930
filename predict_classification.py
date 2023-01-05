import os
import glob
import cv2
import numpy as np
import tensorflow as tf


def predict(model, image, image_size=(120, 120)):
    image = cv2.resize(image, dsize=image_size, interpolation=cv2.INTER_CUBIC)
    img_array = np.asarray(image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    # print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)],
    #                                                                                       100 * np.max(score)))
    return np.argmax(score)


if __name__ == '__main__':
    # config
    class_names = ['1200', '930']
    # load model
    image_dir = "/home/hungtooc/Downloads/samples/private-test-04/1200_object"
    classify_model = tf.keras.models.load_model('/weights/model_v2_gray.h5')
    print(len(glob.glob(image_dir + "/*.jpg")))

    # infer model
    for image_path in glob.glob(image_dir + "/*.jpg"):
        image = cv2.imread(image_path)
        label_id = predict(model=classify_model, image=image)
        os.rename(image_path, f"{image_path.replace('.jpg', f'_{label_id}.jpg')}")
        print(f"image class {label_id}")
