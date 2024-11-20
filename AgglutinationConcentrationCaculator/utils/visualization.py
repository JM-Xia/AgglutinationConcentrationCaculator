import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import joblib


class SingleImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Agglutination Pattern Analysis Software")
        self.root.geometry("800x500")

        # Initialize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check and load trained models
        try:
            # Make sure the dir is existed
            model_dir = 'trained_models'
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Directory '{model_dir}' not found")

            # Set up the path
            self.feature_extractor_path = os.path.join(model_dir, 'feature_extractor.pth')
            self.rf_model_path = os.path.join(model_dir, 'rf_model.joblib')
            self.scaler_path = os.path.join(model_dir, 'scaler.joblib')

            # Check existence
            missing_files = []
            for path in [self.feature_extractor_path, self.rf_model_path, self.scaler_path]:
                if not os.path.exists(path):
                    missing_files.append(path)

            if missing_files:
                raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")

            # Load feature extractor
            print("Loading feature extractor...")
            self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_extractor.fc = torch.nn.Identity()
            self.feature_extractor.load_state_dict(torch.load(self.feature_extractor_path))
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()

            # Load ResRF model and scaler
            print("Loading model and scaler...")
            self.rf_model = joblib.load(self.rf_model_path)
            self.scaler = joblib.load(self.scaler_path)

            # Set up transform
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            print("All models loaded successfully")

            # Set up UI
            self.setup_ui()

        except Exception as e:
            print(f"Error initializing application: {str(e)}")
            messagebox.showerror("Error",
                                 f"Error loading models: {str(e)}\n\n"
                                 "Please ensure:\n"
                                 "1. You have run main.py to train the model\n"
                                 "2. All model files are in the 'trained_models' directory")
            self.root.destroy()
            return

    def setup_ui(self):
        # UI
        title_frame = tk.Frame(self.root)
        title_frame.pack(pady=20)

        title_label = tk.Label(title_frame,
                               text="Agglutination Pattern Analysis Software",
                               font=("Arial", 24))
        title_label.pack()

        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        self.upload_button = tk.Button(button_frame,
                                       text="Upload Image",
                                       command=self.upload_image)
        self.upload_button.pack(side='left', padx=10)

        self.save_button = tk.Button(button_frame,
                                     text="Save Results",
                                     command=self.save_results)
        self.save_button.pack(side='left', padx=10)

        # Results frame
        results_frame = tk.Frame(self.root)
        results_frame.pack(fill='both', expand=True, pady=20)

        self.image_label = tk.Label(results_frame)
        self.image_label.pack(side='left', padx=20)

        self.concentration_label = tk.Label(results_frame,
                                            text="Concentration: Not analyzed",
                                            font=("Arial", 14))
        self.concentration_label.pack(side='right', padx=20)


    def setup_model(self):
        try:
            # Load feature extractor
            self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_extractor.fc = torch.nn.Identity()
            self.feature_extractor.to(self.device)

            # Load feature extractor
            self.feature_extractor.load_state_dict(torch.load('models/feature_extractor_final.pth'))

            # Load RF model and scaler
            self.rf_model = joblib.load('models/rf_model_final.joblib')
            self.scaler = joblib.load('models/scaler_final.joblib')

            # Set up transform
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            print("Models loaded successfully")

        except Exception as e:
            print(f"Error loading models: {e}")
            messagebox.showerror("Error", "Could not load trained models")
            self.root.destroy()

    def setup_ui(self):
        title_label = tk.Label(self.root, text="Agglutination Pattern Analysis Software",
                               font=("Arial", 24))
        title_label.pack(pady=20)

        self.upload_button = tk.Button(self.root, text="Upload image",
                                       command=self.upload_image)
        self.upload_button.pack(pady=20)

        self.save_button = tk.Button(self.root, text="Save results",
                                     command=self.save_results)
        self.save_button.pack(pady=20)

        self.concentration_label = tk.Label(self.root, text="Concentration: Not analyzed",
                                            font=("Arial", 14))
        self.concentration_label.pack(side="right", padx=20)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(side="left", padx=20)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])

        if file_path:
            try:
                self.image_path = file_path
                # Show images
                image = Image.open(file_path)
                image = image.resize((250, 250), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.image_label.config(image=photo)
                self.image_label.image = photo

                # Predict
                dataset = SingleImageDataset(file_path, self.transform)
                loader = DataLoader(dataset, batch_size=1)

                self.feature_extractor.eval()
                with torch.no_grad():
                    for inputs, _ in loader:
                        inputs = inputs.to(self.device)
                        features = self.feature_extractor(inputs)
                        features = features.view(features.size(0), -1).cpu().numpy()

                # Scale features and predict
                features_scaled = self.scaler.transform(features)
                prediction = self.rf_model.predict(features_scaled)[0]

                # Update display
                self.concentration_value = f"{prediction:.2e}"
                self.concentration_label.config(
                    text="Concentration: " + self.concentration_value)

            except Exception as e:
                print(f"Error during prediction: {e}")
                self.concentration_value = "Error"
                self.concentration_label.config(text="Concentration: Error")

    def save_results(self):
        if not hasattr(self, 'image_path'):
            messagebox.showerror("error", "No images uploaded.")
            return

        save_directory = "saved_results"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        base_filename = os.path.basename(self.image_path)
        save_path = os.path.join(save_directory, base_filename)
        image = Image.open(self.image_path)
        image.save(save_path)

        with open(os.path.join(save_directory, "concentration_data.txt"), "a") as file:
            file.write(f"{base_filename}: Concentration = {self.concentration_value}\n")

        messagebox.showinfo("Success", "Results saved successfully")


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()