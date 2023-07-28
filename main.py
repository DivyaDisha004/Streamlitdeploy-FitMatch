# This is a example(clone) Python script.
import numpy as np
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50, preprocess_input
from keras.utils import load_img
from keras.utils import img_to_array
import numpy as np
from numpy.linalg import norm
import os
from tqdm  import tqdm
import pickle



model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


def extract_features(img_path, model):
    img = load_img(img_path,target_size=(224,224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis =0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to t oggle the breakpoint.

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))


features_list = []

for file in tqdm(filenames):
    features_list.append(extract_features(file,model))


pickle.dump(features_list,open('embeddings.pkl' , 'wb'))
pickle.dump(filenames,open('filenames.pkl' , 'wb'))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    #print(model.summary())
    #print(os.listdir('images'))
    #print(len(filenames))
    #print(filenames[0:5])
    # print(np.array(features_list).shape)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
