import os

import riseml


def identity(input_image):
    return  input_image

riseml.serve(identity, port=os.environ.get('PORT'))
