from PIL import ImageOps
import numpy as np

class Preprocessor:
    def __init__(self, resize=False, new_size = None, grayscale=False):
        self.resize = resize
        self.grayscale = grayscale
        self.new_size = new_size
    
    def __call__(self, state):
        res = []
        for image in state:
            if self.grayscale:
                image = ImageOps.grayscale(image)
            if self.resize:
                image = image.resize(self.new_size)            
            res.append(np.asarray(image) / 255)
        return res
            