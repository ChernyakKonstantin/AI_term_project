# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
import matplotlib.pyplot as plt
from net import make_model

model = make_model()
model.load_weights(f'{10000}_frame_tgt_model')

# filters = [layer.get_weights() for layer in model.layers]
# print(len(filters[0]))
weights = [layer.get_weights() for layer in model.get_layer('time_distributed').layer.layers() if 'conv' in layer.name]
print(len(weights))
