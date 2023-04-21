import os
import random
import json

# Set the paths for the R2N2 dataset folders and the splits file
r2n2_dir = './r2n2_shapenet_dataset/r2n2/ShapeNetRendering'
splits_file = './r2n2_shapenet_dataset/splits.json'

# Set the percentage of models to use for testing
test_percentage = 0.2

# Initialize the splits dictionary
splits = {}
splits["train"] = {}
splits["test"] = {}

# Choose the categories to use

# {
    # "04256520": "sofa",
    # "02933112": "cabinet",
    # "02828884": "bench",
    # "03001627": "chair",
    # "03211117": "display",
    # "04090263": "rifle",
    # "03691459": "loudspeaker",
    # "03636649": "lamp",
    # "04401088": "telephone",
    # "02691156": "airplane",
    # "04379243": "table",
    # "02958343": "car",
    # "04530566": "watercraft"
# }

# categories = ["02933112", "03001627", "04379243", "02958343" , "03636649"]

categories = ["03001627", "02933112", "03636649"]

# for category in os.listdir(r2n2_dir): if you want all categories

for category in categories:
    category_dir = os.path.join(r2n2_dir, category)
    if os.path.isdir(category_dir):
        # Get a list of all model IDs in this category
        model_ids = []
        for model_name in os.listdir(category_dir):
            if os.path.isdir(os.path.join(category_dir, model_name)):
                model_ids.append(model_name)
        
        # Shuffle the model IDs and split them into train and test sets
        random.shuffle(model_ids)
        split_index = int(len(model_ids) * (1 - test_percentage))
        train_model_ids = model_ids[:split_index] 
        test_model_ids = model_ids[split_index:]
        
        splits["train"][category] = {}
        splits["test"][category] = {}
        for model_id in train_model_ids:
            splits["train"][category][model_id] = list(range(9))
        
        for model_id in test_model_ids:
            splits["test"][category][model_id] = list(range(9))
            
# Save the splits dictionary to the splits file
with open(splits_file, 'w') as f:
    json.dump(splits, f)