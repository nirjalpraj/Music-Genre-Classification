import sys
import math

import streamlit.web.cli as stcli
import streamlit as st
from streamlit import runtime

runtime.exists()

import numpy as np

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

import librosa

import numpy as np


# Checking the availability of accelerator
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# model for loading the model
model = None


# Necessary constants
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_SEGMENT = 10
DURATION = 30
N_MFCC = 13

# For keeping the size of all data same
sample_per_track = SAMPLE_RATE * DURATION

num_sample_per_segment = int(sample_per_track / N_SEGMENT)
expected_num_mfcc_vectors_per_segment = math.ceil(num_sample_per_segment / HOP_LENGTH)

# model parameters
input_shape = (1, 130, 13)
n_classes = 10
model_file = "./models/classifier.pth"

categorical_code = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        """
        This function initializes the parameters for a maxpool layer

        Parameters
        ------------
        kernel_size : int
        window height and width for the maxpooling window

        stride : int
        the stride of the window. Default value is kernel_size

        padding: int
        implicit zero padding to be added on both sides
        """
        super(MaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, x):
        """
        This function performs max-pool operation on the input

        Parameters
        ------------
        x : tensor, float32
        Input image to the convolution layer

        Returns
        ------------
        x : tensor, float32
        max-pooled output from the last layer
        """
        x = F.max_pool2d(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )

        return x


class BatchNormalization(nn.Module):
    def __init__(self, num_channels):
        super(BatchNormalization, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        # Reshape the input if it's not in 4D (batch_size, channels, height, width)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add a singleton dimension for channels

        # Perform batch normalization
        x = self.batch_norm(x)

        return x


class Reshape(nn.Module):
    """
    Reshape the input to target shape
    """

    def __init__(self, target_shape):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, inputs):
        return inputs.reshape((-1, *self.target_shape))


class Classifier(nn.Module):
    """Defining the architecture of the model."""

    def __init__(self, input_shape, num_classes):
        super(Classifier, self).__init__()

        self.conv_layers = nn.Sequential(
            # Reshaping the input into (batch_size, channel, n_features, n_mfcc_coefficient)
            Reshape(input_shape),
            # First hidden layer
            # expects the batch_size and outputs 32 dimension of data
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            MaxPool(kernel_size=3, stride=2),
            BatchNormalization(32),
            # Second hidden layer
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            MaxPool(kernel_size=3, stride=2),
            BatchNormalization(32),
            # third hidden layer
            nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            MaxPool(kernel_size=2, stride=2),
            BatchNormalization(32),
        )

        # Calculate the output shape after convolutional layers
        fc_input_dims = self.calculate_conv_output_dims(input_shape)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Linear (fully connected) layer
        self.fc1 = nn.Linear(fc_input_dims, 64)

        # dropout layer with 30%
        self.dropout = nn.Dropout(p=0.3)

        # output layer
        self.output = nn.Linear(64, num_classes)

    # calculate dimension form convolution network
    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv_layers(state)

        return int(np.prod(dims.size()))

    def forward(self, inputs):
        x = self.conv_layers(inputs)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output(x)

        return F.softmax(x, dim=1)


def load_model(model_path):
    # Instantiate your model and load the loaded_state_dict
    # model = Classifier(input_shape=input_shape, num_classes=n_classes)
    model = torch.load(model_file).to(device)

    # Set the model to evaluation mode
    model.eval()

    return model


def pre_process(audio) -> np.ndarray:
    # Variable for storing mfccs
    data = []

    # ADC using librosa library
    signal, sr = librosa.load(audio, sr=SAMPLE_RATE)

    # process segments extracting mfcc and storing data
    for segment in range(N_SEGMENT):
        # Description in progress bar

        # Segmenting the file
        start_sample = num_sample_per_segment * segment
        finish_sample = start_sample + num_sample_per_segment

        mfcc = librosa.feature.mfcc(
            y=signal[start_sample:finish_sample],
            sr=sr,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            n_mfcc=N_MFCC,
        )

        # Transposing the mfcc matrix
        mfcc = mfcc.T

        # store mfcc for segment if it has the expected length
        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            # Storing if condition is met
            data.append(mfcc.tolist())

    return np.asarray(data)


def predict(model, inputs):
    """
    This function predicts the predicted outputs for the given test input sample

    Parameters
    ------------
    inputs : tensor, float32
    test input sample

    Returns:
    -------------
    predicted: tensor, float32
    Predicitons for each class by the model
    """
    # converting numpy float64 to float32
    inputs = inputs.astype(np.float32)

    # converting to tensor
    inputs = torch.from_numpy(inputs).to(torch.float32)

    # predicting the output
    predicted = model(inputs.to(device))

    # Obtaining the label of the predicted output
    predicted = torch.argmax(predicted, dim=-1).cpu().numpy()[0]

    return predicted


def main():
    # loading the model
    model = load_model(model_file)

    # Storing predicted value
    predicted = []

    st.markdown(
        """
        <h1 style="text-align: left">Music Genre Classification using CNN</h1>
    """,
        unsafe_allow_html=True,
    )
    st.text(
        """
        This project demonstrates the classification of audion file into different categories of music,
        i.e, "blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"
    """
    )
    st.markdown("---", unsafe_allow_html=True)
    audio = st.file_uploader("Upload Your Audio File", type=["mp3", "wav"])

    if audio:
        st.text("Play Audio")
        st.audio(audio)

        mfccs = pre_process(audio)

        # predict only if a button is clicked
        if st.button("Predict"):
            for each_segment in mfccs:
                prediction = predict(model, each_segment)
                predicted.append(prediction)

            # Calculating the mode of the predicted list
            # find unique values in array along with their counts
            vals, counts = np.unique(predicted, return_counts=True)

            # find mode
            mode_value = np.argwhere(counts == np.max(counts))

            # print list of modes
            # print(vals[mode_value].flatten().tolist())

            predicted_label = vals[mode_value].flatten().tolist()[0]

            st.markdown("---")

            st.markdown(
                f"""
                    <div style='background-color: #080808; padding: 10px; border-radius: 10px'>
                        <h4 style='text-align: left; color: #fff'>The predicted genre of the music is: </h4>
                        <h2 style='text-align: center; color: orangered;'>{categorical_code[predicted_label].upper()}</h2>
                    </div>
                    """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
