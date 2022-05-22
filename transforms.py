import numpy as np

def transpose(img):
  """
  Iterates through the layers of the image and transposes each layer.

  Args:
      img (_type_): _description_

  Returns:
      _type_: _description_
  """
  layers = img.shape[-1]
  return np.stack([img[:,:,l].T for l in range(layers)], axis=2)

def horizontal_flip(img):
  """
  Iterates through the layers of the image and flips each layer alongside the horizontal axis.

  Args:
      img (_type_): _description_

  Returns:
      _type_: _description_
  """
  layers = img.shape[-1]
  return np.stack([[row[::-1] for row in img[:,:,l]] for l in range(layers)], axis=2)

def vertical_flip(img):
  """
  Iterates through the layers of the image and flips each layer alongside the vertical axis.

  Args:
      img (_type_): _description_

  Returns:
      _type_: _description_
  """
  layers = img.shape[-1]
  return np.stack([img[:,:,l][::-1] for l in range(layers)], axis=2)