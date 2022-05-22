import torch

classes = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

label_classes = {label:num for num, label in enumerate(classes)} # mapping labels to numbers
label_classes_back_mapping = {v:k for k,v in label_classes.items()} # mapping numbers to labels

torch_mapping = lambda label: torch.tensor((label_classes[label]), device=torch.device("cuda"))	 # mapping labels to tensors