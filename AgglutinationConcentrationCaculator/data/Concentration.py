import os
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ConcentrationDataset(Dataset):
    def __init__(self, img_dir, transform=None, val_split=0.2, is_train=True):
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train  # Use boolean flag to differentiate between train and val splits.

        all_imgs = os.listdir(self.img_dir)
        train_imgs, val_imgs = train_test_split(all_imgs, test_size=val_split, random_state=42)

        self.dataset_imgs = train_imgs if self.is_train else val_imgs

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
            # Handle the None case, e.g., by logging and skipping or assigning a default value
            concentration = 0  # Example default value
            print(f"Warning: Assigning default concentration for {img_name}")

        return image, concentration

    def get_all_concentrations(self):
        # This method returns a list of all concentration values in the dataset
        concentrations = []
        for img_name in self.dataset_imgs:
            concentration = self.parse_concentration(img_name)
            if concentration is not None:
                concentrations.append(concentration)
        return concentrations

    def parse_concentration(self, img_name):
        # Split the filename to extract the concentration part
        parts = img_name.split('_')[0]  # Assuming concentration is the first part

        # Handle scientific notation with '^'
        if '^' in parts:
            base, exponent = parts.split('^')
            try:
                base = float(base)
                exponent = int(exponent.split()[0])  # Split in case there's additional info like "(4)"
                return base ** exponent
            except ValueError:
                print(f"Could not parse concentration from image name {img_name}")
                return None
        # Handle negative exponents with '-'
        elif '-' in parts:
            try:
                base, exponent = parts.split('-')
                if base == "10":
                    exponent = int(exponent.split()[0])  # Handle additional info in parentheses
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
