import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from skimage import measure
from scipy.ndimage import label as scipy_label
import tifffile as tiff

class OpenSARShipDataset(Dataset):
    """
    PyTorch Dataset for OpenSARShip
    
    Dataset structure:
    - PATCH/: Original SAR patches
    - PATCH_CAL/: Calibrated patches  
    - PATCH_RGB/: RGB visualization
    - PATCH_UINT8/: 8-bit converted patches
    - ais.csv: Metadata with AIS information
    """
    
    def __init__(self, root_dir):
        """
        Args:
            root_dir: Path to OpenSARShip dataset
            split: 'train', 'val', or 'test'
            transform: Image transformations
            target_classes: Ship classes to use
            extract_features: Whether to extract scale-variant features
            target_size: Target image size (64x64)
        """
        self.root_dir = Path(root_dir)
        
        # Load AIS metadata
        self.ais_data = pd.read_csv(self.root_dir / 'metadata.csv')
        
        # Map ship types to our 4 classes
        self.class_mapping = {
            'Cargo': 0,      # Bulk Carrier
            'Container': 1,       # Container Ship (we'll refine this)
            'Fishing': 2,    # Fishing Vessel
            'Tanker': 3,     # Tanker
        }
        
        # Filter for target classes
        self.filter_data()
        
    def filter_data(self):
        """Filter dataset for 4 target ship classes"""
        # OpenSARShip uses different naming - map to paper's classes
        # Bulk Carrier, Container Ship, Fishing, Tanker
        
        # Filter by ship type
        ship_type_col = 'category'
        
        valid_indices = []
        filtered_data = []
        label_dict = {0:0, 1:0, 2:0, 3:0}
        print(self.ais_data[ship_type_col].value_counts())
        
        for idx, row in self.ais_data.iterrows():
            ship_type = row[ship_type_col]
            
            # Map to our 4 classes
            if 'Cargo' in str(ship_type):
                elaborated_type = row['Elaborated_type']
                if str(elaborated_type) == "Bulk Carrier":
                    label = 0  # Bulk Carrier
                elif str(elaborated_type) == "Container Ship":
                    label = 1  # Container Ship
                else:
                    continue  # Skip other cargo types

            elif str(ship_type) == 'Fishing':
                label = 2  # Fishing
            elif 'Tanker' in str(ship_type):
                label = 3  # Tanker
            else:
                continue

            img_path = f"resized_new/{row['patch_cal']}"
            label_dict[label] += 1
            
            if Path(img_path).exists():
                valid_indices.append(idx)
                filtered_data.append({
                    'label': label,
                    'img_path': img_path,   
                    'incidence': row['Incidence'],
                })
        
        self.data = filtered_data
        print(f"Loaded {len(self.data)} samples")
        print(f"Class distribution:\n{label_dict}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        row = self.data[idx]
        
        # Load image and get features
        if "image" not in row:
            image = tiff.imread(str(row['img_path'])).astype(np.float32)
            features = self.extract_scale_variant_features(image, row.get('incidence', 0))
            self.data[idx]['image'] = image
            self.data[idx]['features'] = features

        return {
            'image': self.data[idx]['image'],
            'img_path': row['img_path'],
            'features': self.data[idx]['features'],
            'label': row['label']
        }
    
    def get_labels(self):
        """Return all labels for stratified splitting"""
        return [self.data[i]['label'] for i in range(len(self.data))]
    
    def extract_scale_variant_features(self, stacked_image, incidence_angle):
        """
        Extract 14 scale-variant features as described in Table 2
        Uses original size images before resizing
        """
        features = np.zeros(14, dtype=np.float32)
        
        # 1. Angle of incidence
        features[0] = incidence_angle
        
        # Ship detection using CA-CFAR (Eq. 2, 3)
        vh_img = stacked_image[:,:,0]
        binary_mask = self.ca_cfar_detection(vh_img)
        
        # Get region properties
        labeled_img, num_features_detected = scipy_label(binary_mask)

        if num_features_detected > 0:
            props = measure.regionprops(labeled_img, intensity_image=stacked_image) 
            
            # Take largest region as ship
            ship = max(props, key=lambda x: x.area)
            
            # 2. Length (from minimum bounding rectangle)
            y0, x0, y1, x1 = ship.bbox
            features[1] = max(y1 - y0, x1 - x0)
            
            # 3. Total pixels
            features[2] = ship.area
            
            # 4. Mass (sum of intensity values)
            features[3] = vh_img[labeled_img == ship.label].sum()
            
            # 5. Area of minimum bounding rectangle
            features[4] = ship.bbox_area
            
            # 6-7. Significance (Eq. 4)
            ship_pixels = vh_img[labeled_img == ship.label].astype(np.float32)
            background_pixels = vh_img[labeled_img != ship.label].astype(np.float32)

            m_t = ship_pixels.mean()
            m_max = ship_pixels.max()
            m_b = background_pixels.mean()
            s_b = background_pixels.std()

            if len(background_pixels) > 0:
                if s_b > 0:
                    features[5] = (m_t - m_b) / s_b  # significance_mean
                    features[6] = (m_max - m_b) / s_b   # significance_max
            
            # 8-14. Hu moments (7 invariant moments)
            hu_moments = ship.moments_hu
            features[7:14] = hu_moments
        
        return features

    def ca_cfar_detection(self, image, pfa=1e-3):
        """
        CA-CFAR ship detection as per Eq. (2) and (3)
        """
        N = image.size
        alpha = N * (pfa ** (-1/N) - 1)
        
        background_mean = image.mean()
        threshold = alpha * background_mean
        
        binary_mask = image > threshold
        return binary_mask
    

class FinalDataset(Dataset):
    def __init__(self, csv_path):
        """
        Load pre-augmented dataset from CSV
        
        Args:
            csv_path: Path to CSV with image paths and features
            mean: Training set mean
            std: Training set std
        """
        self.data = pd.read_csv(csv_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image (already standardized)
        image = tiff.imread(str(row['img_path'])).astype(np.float32)
        image = torch.from_numpy(image).float()
        if image.ndim == 3 and image.shape[-1] == 2:
            image = image.permute(2, 0, 1)  # [64, 64, 2] -> [2, 64, 64]
        
        # Load features (stored as string in CSV)
        features = np.array(eval(row['features']), dtype=np.float32)
        
        return {
            'image': image,
            'features': torch.from_numpy(features).float(),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }