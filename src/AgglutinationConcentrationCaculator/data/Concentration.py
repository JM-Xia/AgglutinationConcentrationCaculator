import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

"""
Dataset class for loading and processing agglutination concentration images.

The dataset splits data into training , testing and validation sets, and handles image loading
and concentration value parsing from filenames.
"""

class ConcentrationDataset(Dataset):
    def __init__(self, img_dir, transform=None, split='train'):
        self.img_dir = img_dir
        self.transform = transform

        # Get all image files
        all_imgs = os.listdir(self.img_dir)

        # First split: 70% train, 30% temp
        train_imgs, temp_imgs = train_test_split(
            all_imgs,
            train_size=0.7,
            random_state=42
        )

        # Second split: Split temp into equal val and test
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=0.5,
            random_state=42
        )

        # Select appropriate split
        if split == 'train':
            self.dataset_imgs = train_imgs  # 70%
        elif split == 'val':
            self.dataset_imgs = val_imgs  # 15%
        else:  # test
            self.dataset_imgs = test_imgs  # 15%


    def __len__(self):
        return len(self.dataset_imgs)

    def __getitem__(self, idx):
        img_name = self.dataset_imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        concentration = self.parse_concentration(img_name)
        if concentration is None:
            concentration = 0
            print(f"Warning: Assigning default concentration for {img_name}")

        return image, concentration

    def get_all_concentrations(self):
        concentrations = []
        for img_name in self.dataset_imgs:
            concentration = self.parse_concentration(img_name)
            if concentration is not None:
                concentrations.append(concentration)
        return concentrations

    def parse_concentration(self, img_name):
        # Split the filename to extract the concentration part
        parts = img_name.split('_')[0]

        # Handle scientific notation with '^'
        if '^' in parts:
            base, exponent = parts.split('^')
            try:
                base = float(base)
                exponent = int(exponent.split()[0])
                return base ** exponent
            except ValueError:
                print(f"Could not parse concentration from image name {img_name}")
                return None
        # Handle negative exponents with '-'
        elif '-' in parts:
            try:
                base, exponent = parts.split('-')
                if base == "10":
                    exponent = int(exponent.split()[0])
                    return 10 ** (-int(exponent))
                else:
                    print(f"Unexpected base found in scientific notation: {img_name}")
                    return None
            except ValueError:
                print(f"Could not parse concentration from image name {img_name}")
                return None
        else:
            try:
                # Direct conversion if no scientific notation
                return float(parts)
            except ValueError:
                print(f"Could not convert concentration to float for image: {img_name}")
                return None
