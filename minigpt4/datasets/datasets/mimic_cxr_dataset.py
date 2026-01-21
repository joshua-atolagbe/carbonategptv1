import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset

class MimicCxrDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)
        
        self.instruction_pool = [
            'Describe this image in detail',
            'Take a look at this image and describe what you notice',
            'Please provide a detailed description of the picture',
            'Could you describe the contents of this image for me?'
        ]
        
        # Validate dataset and remove missing files
        self._validate_dataset()

    def _validate_dataset(self):
        """Remove entries with missing images"""
        valid_ann = []
        missing_count = 0
        
        print("Validating dataset...")
        for idx, item in enumerate(self.ann):
            image_path = os.path.join(self.vis_root, item['image_path'])
            if os.path.exists(image_path):
                valid_ann.append(item)
            else:
                missing_count += 1
                if missing_count <= 10:  # Print first 10 missing files
                    print(f"Missing: {image_path}")
        
        if missing_count > 10:
            print(f"... and {missing_count - 10} more missing files")
        
        print(f"\nDataset validation complete:")
        print(f"  Total samples: {len(self.ann)}")
        print(f"  Missing files: {missing_count}")
        print(f"  Valid samples: {len(valid_ann)}")
        
        self.ann = valid_ann

    def load_image(self, image_id):
        """Load image with error handling"""
        try:
            image_path = os.path.join(self.vis_root, image_id)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                return None
                
            grayscale_image = Image.open(image_path).convert("L")
            grayscale_image = grayscale_image.resize((448, 448))
            image = Image.new("RGB", grayscale_image.size)
            image.paste(grayscale_image)
            image = self.vis_processor(image)
            return image
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # Try to get a valid sample
        max_attempts = 10  # Prevent infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            try:
                info = self.ann[index]
                image = self.load_image(info['image_path'])
                
                # If image loading failed, try next sample
                if image is None:
                    index = (index + 1) % len(self)
                    attempts += 1
                    continue
                
                instruction = random.choice(self.instruction_pool)
                instruction = f'<Img><ImageHere></Img> {self.text_processor(instruction)}'

                return {
                    "image": image,
                    "instruction_input": instruction,
                    "answer": info['caption'],
                    "image_id": info['image_id'],
                }
                
            except Exception as e:
                print(f"Error processing index {index}: {e}")
                index = (index + 1) % len(self)
                attempts += 1
                continue
        
        # If all attempts failed, raise error
        raise RuntimeError(f"Failed to load valid sample after {max_attempts} attempts")

class evalMIMICDataset(Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

        self.instruction_pool = [
            'Describe this image in detail',
            'Take a look at this image and describe what you notice',
            'Please provide a detailed description of the picture',
            'Could you describe the contents of this image for me?'
        ]
        
        # Validate dataset and remove missing files
        self._validate_dataset()

    def _validate_dataset(self):
        """Remove entries with missing images"""
        valid_data = []
        missing_count = 0
        
        print("Validating evalMIMIC dataset...")
        for idx, item in enumerate(self.loaded_data):
            image_path = os.path.join(self.root_path, item['image_path'])
            if os.path.exists(image_path):
                valid_data.append(item)
            else:
                missing_count += 1
                if missing_count <= 10:  # Print first 10 missing files
                    print(f"Missing: {image_path}")
        
        if missing_count > 10:
            print(f"... and {missing_count - 10} more missing files")
        
        print(f"\nevalMIMIC validation complete:")
        print(f"  Total samples: {len(self.loaded_data)}")
        print(f"  Missing files: {missing_count}")
        print(f"  Valid samples: {len(valid_data)}")
        
        self.loaded_data = valid_data

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        info = self.loaded_data[idx]
        img_id = '{}.jpg'.format(info['image_id'])
        image_path = os.path.join(self.root_path, info['image_path'])
        grayscale_image = Image.open(image_path).convert("L")
        grayscale_image = grayscale_image.resize((448, 448))
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)

        answer = info['caption']
        question = random.choice(self.instruction_pool)

        return image, question, img_id
    
    
class evalDetectMimicDataset(Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        
        # Validate dataset and remove missing files
        self._validate_dataset()

    def _validate_dataset(self):
        """Remove entries with missing images"""
        valid_data = []
        missing_count = 0
        
        print("Validating evalDetectMimic dataset...")
        for idx, item in enumerate(self.loaded_data):
            image_path = os.path.join(self.root_path, item['key'])
            if os.path.exists(image_path):
                valid_data.append(item)
            else:
                missing_count += 1
                if missing_count <= 10:  # Print first 10 missing files
                    print(f"Missing: {image_path}")
        
        if missing_count > 10:
            print(f"... and {missing_count - 10} more missing files")
        
        print(f"\nevalDetectMimic validation complete:")
        print(f"  Total samples: {len(self.loaded_data)}")
        print(f"  Missing files: {missing_count}")
        print(f"  Valid samples: {len(valid_data)}")
        
        self.loaded_data = valid_data

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['key']
        sent = data['objects']
        image_path = os.path.join(self.root_path, img_id)
        grayscale_image = Image.open(image_path).convert("L")
        image = Image.new("RGB", grayscale_image.size)
        image.paste(grayscale_image)
        image = self.vis_processor(image)
        question = f"[detection] {sent}"

        return image, question, img_id
