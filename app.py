import json
import numpy as np
import torch
from transformers import pipeline
from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer


class InferlessPythonModel:

    def initialize(self):
        pass

    def infer(self, inputs, stream_output_handler):
        pass


    # perform any cleanup activity here
    def finalize(self,args):
        pass
