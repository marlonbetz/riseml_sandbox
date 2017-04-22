import os
import riseml
from PIL import Image
from io import BytesIO
import numpy as np

def predict(input_image):
    input_image = Image.open(BytesIO(input_image)).convert('RGB')
    image = np.asarray(input_image, dtype=np.float32)

    image = image.transpose(2, 0, 1)
    image = np.tile(np.mean(image,axis=0),reps=(3,1,1))
    image = image.reshape((1,) + image.shape)

    result = np.uint8(image[0].transpose((1, 2, 0)))

    med = Image.fromarray(result)

    output_image = BytesIO()
    med.save(output_image, format='JPEG')
    return output_image.getvalue()

riseml.serve(predict, port=os.environ.get('PORT'))
