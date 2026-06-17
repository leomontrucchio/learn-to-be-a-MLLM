import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class HighResDataset(Dataset):
    def __init__(self, split, class_name, dataset_path, feature_dir=None, target_layers=None):
        self.split = split
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.feature_dir = feature_dir # if not None, we load the features
        self.target_layers = target_layers
        
        base_path = os.path.join(dataset_path, class_name)
        if split == 'train':
            self.img_paths = sorted(glob.glob(os.path.join(base_path, 'train', 'good', '*.png')))
        elif split == 'validation':
            self.img_paths = sorted(glob.glob(os.path.join(base_path, 'validation', 'good', '*.png')))
        elif split == 'test':
            self.img_paths = sorted(glob.glob(os.path.join(base_path, 'test', '**', '*.png'), recursive=True))
            self.img_paths = [p for p in self.img_paths if os.path.isfile(p)]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # If we are in training/val e we have the features
        if self.feature_dir and self.target_layers:
            # Carica esattamente i due layer specificati
            l1 = torch.load(os.path.join(self.feature_dir, self.split, f"{filename}_layer_{self.target_layers[0]}.pt"), map_location='cpu')
            l2 = torch.load(os.path.join(self.feature_dir, self.split, f"{filename}_layer_{self.target_layers[1]}.pt"), map_location='cpu')
            return l1, l2

        # Otherwise, we load the image (to extract features or testing)
        pil_img = Image.open(img_path).convert('RGB')
        if self.split == 'test':
            return pil_img, img_path
        return pil_img, filename

def hd_collate_fn(batch):
    # Feature Loading (Training/Val)
    if isinstance(batch[0][0], torch.Tensor):
        earlier_batch = torch.stack([item[0] for item in batch])
        later_batch = torch.stack([item[1] for item in batch])
        return earlier_batch, later_batch
    
    # Image Loading (Extraction/Inference)
    if isinstance(batch[0], tuple) and isinstance(batch[0][1], str):
        return [item[0] for item in batch], [item[1] for item in batch]
    return batch

def get_hd_data_loader(split, class_name, dataset_path, batch_size=4, feature_dir=None, target_layers=None):
    dataset = HighResDataset(split, class_name, dataset_path, feature_dir=feature_dir, target_layers=target_layers)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        collate_fn=hd_collate_fn
    )
    return loader