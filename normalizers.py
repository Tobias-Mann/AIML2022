import numpy as np


class DistNormalizer:
  def __init__(self, dist):
    """
        Args:
        dist (dict): Layerwise mean and standard deviation of the dataset, in the form of {layer: {"mean": mean, "std": std}} 
    """
    self.dist = dist
  
  def __call__(self, img):
    """
    Call the normalizer on a single image (3d numpy array) and return the normalized image.

    Args:
        img (numpy.arry): 3d Image to be normalized.

    Returns:
        numpy.arry: Normalized image.
    """
    layers = img.shape[-1]
    return np.stack([(img[:,:,int(l)]-self.dist[str(l)]["mean"])/(self.dist[str(l)]["std"]+1e-10 )for l in map(int, range(layers))], axis=2)


class QuantileNormalizer:
  def __init__(self, *args):
    pass
  
  def __call__(self, img):
    layers = img.shape[-1]
    return np.stack([(img[:,:,l]-np.mean(img[:,:,l]))/(0.00001+np.std(img[:,:,l])) for l in range(layers)], axis=2)