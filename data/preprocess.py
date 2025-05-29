import h5py
import numpy as np
from tqdm import tqdm
import os

# Define your noisy long IDs
noisy_ids = {
    "C11_s1_w127B03F50-0445-404D-8686-992152316A15",
    "G02_s2_w17DA4AF6A-01FB-4517-9E5B-CCDD62A1D4B6",
    "Week1_150607_B03_s2_w16BAE18E6-E27B-4683-90FA-581B8ADA6CB1", # maybe
    "Week1_150607_E04_s4_w12BDBC4CA-CCA1-48C8-B4D5-F229EB266410", # maybe
    "Week1_150607_E11_s1_w1F5053E7F-88CD-40F8-AAE0-0633FF5CBBA4", # maybe
    "Week1_150607_E11_s2_w185FD4BFE-407C-43FA-8376-281D21993540", # maybe
    "Week2_180607_E02_s3_w109D7263C-9A3C-4717-B6AA-BA1ECCF45A70", # maybe
    "Week2_180607_F02_s4_w14EBE2925-CBA5-42C4-8853-A78BACC04D3C",
    "Week3_290607_B11_s3_w1E2E13398-A442-40FE-AF91-FB9B55A9D664",
    "Week5_130707_C11_s4_w1EBDD4E60-9887-4A72-BFD1-C3D6EB633BE4",
    "Week5_130707_C11_s4_w116C137AD-BD10-4685-984A-A8E52DA612BA",
    "Week5_130707_D02_s2_w1286A02BA-8BE0-4267-B919-E6223AB4F182",
    "Week6_200607_C11_s3_w19357A9CA-9915-45A9-8F2E-09B6AE811524",
    "Week6_200607_E02_s2_w13ED36977-F80B-40C5-8151-7CF6DB9782AE",
    "Week6_200607_E11_s4_w1AABB4630-0530-43D7-B915-902AA6480CC3",
    "Week7_230707_C06_s2_w1D292D8D5-F5A6-4CC9-9A28-D4713C8D8076",
    "Week7_230707_E02_s2_w182895F08-11E3-4AB6-B2B8-1BEECE2862F4",
    "Week7_230707_F02_s1_w176094EF3-CBD9-4F0C-8388-D0A55B172980",
    "Week7_230707_G02_s2_w1B5CBE42C-5E9F-4F2C-8094-23EF0BF691AA",
    "Week7_230707_G02_s3_w1D044621A-12B7-4CCC-AD48-B4EFC5C0D221",
    "Week8_4sites_B02_s2_w14F7C0A4C-02B2-49C5-909F-37CB44AD0FDF", # maybe
    "Week8_4sites_C09_s2_w14A741543-3F34-49DE-A541-EB80EA9D937D",
    "Week8_4sites_E11_s3_w13E4113DB-0EB6-4ED5-BB6E-1E3EE3F5EE57",
    "Week9_090907_C02_s2_w1A5369617-A74F-4F39-BF1F-D5834D20B5C0",
    "Week9_090907_C11_s3_w195438B0C-C73B-471B-AF96-85FA31D58F37", # maybe
}

input_path = 'D:/BBBC021/BBBC021_dataset.h5'
output_path = 'D:/BBBC021/BBBC021_cleaned_preprocessed.h5'

# Preprocessing functions
def crop_image(image: np.ndarray, pix: int = 2) -> np.ndarray:
    return image[pix:-pix, pix:-pix, :] if image.shape == (68, 68, 3) else None

def normalize_image(image: np.ndarray, method: str) -> np.ndarray:
    if method == "min_max":
        image = image.astype(np.float32)
        min_vals = image.min(axis=(0, 1), keepdims=True)
        max_vals = image.max(axis=(0, 1), keepdims=True)
        range_vals = np.maximum(max_vals - min_vals, 1e-5)
        return (image - min_vals) / range_vals
    elif method == "max":
        image = image.astype(np.float32)
        max_vals = image.max(axis=(0, 1), keepdims=True)  # shape (1, 1, C)
        max_vals = np.maximum(max_vals, 1e-5)  # prevent division by 0
        return image / max_vals
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def is_noisy(name: str, bad_ids: set) -> bool:
    prefix = '_'.join(name.split('_')[:-1])  # Remove the last `_###` part
    return prefix in bad_ids

# Processing
with h5py.File(input_path, 'r') as in_f, h5py.File(output_path, 'w') as out_f:
    raw_images = in_f['images']
    raw_names = in_f['image_names']
    total = len(raw_images)

    img_h = raw_images.shape[1]  # Height of the images
    img_w = raw_images.shape[2]  # Width of the images
    crop_pixels = 0  # Number of pixels to crop from each side

    img_shape = (3, img_h - 2 * crop_pixels, img_w - 2 * crop_pixels)  # (C, H, W)
    img_dtype = np.float32

    # Create resizable datasets in output file
    image_dataset = out_f.create_dataset(
        'images',
        shape=(0, *img_shape),
        maxshape=(None, *img_shape),
        dtype=img_dtype,
        chunks=(1, *img_shape),
        compression='gzip'
    )

    name_dataset = out_f.create_dataset(
        'image_names',
        shape=(0,),
        maxshape=(None,),
        dtype=h5py.string_dtype(),
        compression='gzip'
    )

    index = 0
    for i in tqdm(range(total), desc="Processing Images"):
        name = raw_names[i].decode('utf-8')
        if is_noisy(name, noisy_ids):
            print(f"Removed image: {name}")
            continue

        img = raw_images[i]
        if crop_pixels > 0:
            img = crop_image(img, crop_pixels)

        if img is None:
            raise ValueError(f"Image {name} has unexpected shape: {raw_images[i].shape}")

        img = normalize_image(img, "min_max") # or "max"
        img = np.transpose(img, (2, 0, 1))

        # Save to output dataset
        image_dataset.resize(index + 1, axis=0)
        name_dataset.resize(index + 1, axis=0)

        image_dataset[index] = img
        name_dataset[index] = name.encode('utf-8')
        index += 1

print(f"\nSaved {index} cleaned & preprocessed images to: {output_path}")
