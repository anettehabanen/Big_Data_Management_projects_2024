import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from fedn.utils.helpers.helpers import get_helper
#import tensorflow as tf
#from tensorflow import keras

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def save_parameters(model, out_path):
    """Save model paramters to file.

    :param model: The model to serialize.
    :type model: torch.nn.Module
    :param out_path: The path to save to.
    :type out_path: str
    """
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)


"""def create_model(input_shape=(32, 32, 3), dimension='VGG16'):
    num_classes = 10
    model = Sequential()
    model.add(keras.Input(shape=input_shape))
    for x in cfg[dimension]:
        if x == 'M':
            model.add(MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(Conv2D(x, (3, 3), padding='same', trainable=True))
            model.add(BatchNormalization(trainable=True))
            model.add(Activation(activations.relu))

    # model.add(Flatten())
    model.add(AveragePooling2D(pool_size=(1, 1)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                 optimizer=opt, metrics=['accuracy'])

    return model"""

def compile_model():
    model = models.vgg16()
    input_lastLayer = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(input_lastLayer,10)
    return model


def load_parameters(model_path):
    """Load model parameters from file and populate model.

    param model_path: The path to load from.
    :type model_path: str
    :return: The loaded model.
    :rtype: torch.nn.Module
    """
    model = compile_model()
    parameters_np = helper.load(model_path)

    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict({key: torch.tensor(x) for key, x in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


def init_seed(out_path="seed.npz"):
    """Initialize seed model and save it to file.

    :param out_path: The path to save the seed model to.
    :type out_path: str
    """
    # Init and save
    model = compile_model()
    save_parameters(model, out_path)


if __name__ == "__main__":
    init_seed("../seed.npz")
