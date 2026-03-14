import torch
import nibabel as nib
import numpy as np
from typing import Union, Tuple, Optional
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    RandShiftIntensityd,
    RandFlipd,
    EnsureTyped,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm

class MRIProcessor:
    """
    Research-grade MRI Processor for 3D Neuroimaging.
    Provides methods for loading, normalization, and deep learning-based skull stripping.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.preprocessing_pipeline = Compose([
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            NormalizeIntensityd(keys=["image"]),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            EnsureTyped(keys=["image"]),
        ])

    def load_volume(self, file_path: str) -> np.ndarray:
        """
        Loads a NIfTI MRI volume.

        Args:
            file_path: Path to the .nii or .nii.gz file.

        Returns:
            A numpy array of the MRI volume.
        """
        img = nib.load(file_path)
        data = img.get_fdata()
        return data

    def preprocess_volume(self, data: np.ndarray) -> torch.Tensor:
        """
        Applies standard preprocessing pipeline.

        Args:
            data: Input numpy array (H, W, D).

        Returns:
            Preprocessed torch Tensor.
        """
        # Add channel and batch dimensions
        input_data = {"image": data[np.newaxis, ...]}
        processed = self.preprocessing_pipeline(input_data)
        return processed["image"].to(self.device)

    def extract_brain_mask(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for a deep learning based skull stripping (brain extraction).
        In a real scenario, this would load a pre-trained model (e.g., HD-BET or SynthStrip).

        Args:
            volume: 4D Tensor (C, H, W, D).

        Returns:
            Binary brain mask.
        """
        # Example logic: Simple thresholding as a fallback for the placeholder
        mask = (volume > 0.1).float()
        return mask

    def get_model(self, in_channels: int = 1, out_channels: int = 2) -> torch.nn.Module:
        """
        Initializes a 3D UNet for neuroimaging tasks (e.g., segmentation).
        """
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(self.device)

if __name__ == "__main__":
    # Example usage
    processor = MRIProcessor()
    print(f"MRIProcessor initialized on device: {processor.device}")
