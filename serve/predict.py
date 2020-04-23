import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import re
from bs4 import BeautifulSoup

from model import LSTMClassifier

from utils import review_to_words, convert_and_pad


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    print(prediction_output)
    # return str(prediction_output.item())
    return str(prediction_output)


def predict_fn(input_data, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')

    # TODO: Process input_data so that it is ready to be sent to our model.
    #       You should produce two variables:
    #         data_X   - A sequence of length 500 which represents the converted review
    #         data_len - The length of the review

    data_X = None
    data_len = None
    input_data_words = review_to_words(input_data)
    data_X, data_len = convert_and_pad(model.word_dict, input_data_words)
    # data_X = pd.concat([pd.DataFrame(test_data_len), pd.DataFrame(test_data)], axis=1)

    # Using data_X and data_len we construct an appropriate input tensor. Remember
    # that our model expects input data of the form 'len, review[500]'.
    # data_pack = np.hstack((data_len, data_X))
    data_stacked = np.hstack((data_len, data_X))

    data_stacked = data_stacked.reshape(1, -1)

    data = torch.from_numpy(data_stacked)
    data = data.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    # TODO: Compute the result of applying the model to the input data. The variable `result` should
    #       be a numpy array which contains a single integer which is either 1 or 0

    with torch.no_grad():
        output = model.forward(data)

    result = np.round(output.numpy())
    # result = predictor.predict(data.values)

    return result


def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()

    text = BeautifulSoup(review, "html.parser").get_text() 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    words = text.split() 
    words = [w for w in words if w not in stopwords.words("english")] 
    words = [PorterStemmer().stem(w) for w in words]  

    return words


def convert_and_pad(word_dict, sentence, pad=500):
    no_word = 0
    missing_word = 1

    working_sentence = [no_word] * pad

    for index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[index] = word_dict[word]
        else:
            working_sentence[index] = missing_word

    return working_sentence, min(len(sentence), pad)


def convert_and_pad_data(word_dict, data, pad=500):
    result = []
    lengths = []

    for sentence in data:
        converted, lengths = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(lengths)

    return np.array(result), np.array(lengths)
