import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import sys
import os
import tempfile
import zipfile
import io
import re
from PIL import Image, __version__ as PIL_VERSION
import warnings
import base64
import time
import threading

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#-------------------------------------------------------------------
# Page Configuration
#-------------------------------------------------------------------
st.set_page_config(page_title="SAM Plus/ VNC-Dist", layout="wide")

#-------------------------------------------------------------------
# Inactivity Timer Setup (applies to the entire app)
#-------------------------------------------------------------------
TIMEOUT_SECONDS = 300  # Auto shutdown after 5 minutes of inactivity

def inactivity_monitor():
    """Background thread that checks for inactivity and exits if TIMEOUT_SECONDS elapsed."""
    while True:
        time.sleep(10)
        if time.time() - st.session_state.get("last_interaction", time.time()) > TIMEOUT_SECONDS:
            st.write(f"No activity detected for {TIMEOUT_SECONDS} seconds. Shutting down.")
            sys.exit(0)

if "last_interaction" not in st.session_state:
    st.session_state["last_interaction"] = time.time()
if "inactivity_thread" not in st.session_state:
    thread = threading.Thread(target=inactivity_monitor, daemon=True)
    st.session_state["inactivity_thread"] = thread
    thread.start()
st.session_state["last_interaction"] = time.time()

#-------------------------------------------------------------------
# Helper Functions for SAM Plus Tab (Logos and Centered Text)
#-------------------------------------------------------------------
def load_resize_encode_image(image_path, scale_factor):
    try:
        image = Image.open(image_path)
        original_width, original_height = image.size
        new_size = (max(1, int(original_width * scale_factor)), max(1, int(original_height * scale_factor)))
        resized_image = image.resize(new_size, resample=Image.LANCZOS)
        buffered = io.BytesIO()
        resized_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return img_base64
    except Exception as e:
        st.sidebar.error(f"Error loading image {os.path.basename(image_path)}: {e}")
        return None

def display_side_by_side_logos(logo1_path, logo2_path, scale_factor1=0.2, scale_factor2=0.24):
    logo1_base64 = load_resize_encode_image(logo1_path, scale_factor1)
    logo2_base64 = load_resize_encode_image(logo2_path, scale_factor2)
    if logo1_base64 and logo2_base64:
        st.sidebar.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <img src="data:image/png;base64,{logo1_base64}" style="height: auto; max-width: 45%;">
                <img src="data:image/png;base64,{logo2_base64}" style="height: auto; max-width: 45%;">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if not logo1_base64:
            st.sidebar.warning("First logo image not found. Please ensure the file path is correct.")
        if not logo2_base64:
            st.sidebar.warning("Second logo image not found. Please ensure the file path is correct.")

def display_centered_text(text):
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; font-weight: bold; color: black;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

#-------------------------------------------------------------------
# Functions to Load SAM Model and Process Images (SAM Plus Tab)
#-------------------------------------------------------------------
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

@st.cache_resource
def load_sam_model(checkpoint_path, model_type="", device=""):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=2,
        pred_iou_thresh=0.98,
        stability_score_thresh=0.98,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=99000,
    )
    return mask_generator

def process_image(image, mask_generator):
    masks = mask_generator.generate(image)
    mask_images = []
    for mask_info in masks:
        mask_array = mask_info["segmentation"]
        mask_array = (mask_array > 0).astype(np.uint8) * 255
        inverted_mask = cv2.bitwise_not(mask_array)
        mask_smoothed = cv2.GaussianBlur(inverted_mask, (21, 21), 0)
        mask_smoothed = cv2.medianBlur(mask_smoothed, 33)
        mask_smoothed = cv2.bilateralFilter(mask_smoothed, 9, 75, 75)
        erosion_kernel = np.ones((19, 19), np.uint8)
        mask_smoothed = cv2.erode(mask_smoothed, erosion_kernel, iterations=1)
        closing_kernel = np.ones((21, 21), np.uint8)
        mask_smoothed = cv2.morphologyEx(mask_smoothed, cv2.MORPH_CLOSE, closing_kernel)
        mask_rgba = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
        mask_rgba[:, :, 3] = 0
        mask_rgba[mask_smoothed > 0] = [0, 0, 0, 255]
        mask_rgba[:25, :, 3] = 0
        mask_rgba[-25:, :, 3] = 0
        mask_rgba[:, :25, 3] = 0
        mask_rgba[:, -25:, 3] = 0
        mask_images.append(mask_rgba)
    return mask_images

#-------------------------------------------------------------------
# Functions for Manual Segmentation Tab (TIFF to PNG Conversion)
#-------------------------------------------------------------------
def convert_tiff_to_png(tiff_file, output_buffer):
    try:
        # Open the TIFF image and convert to RGBA.
        img = Image.open(tiff_file).convert("RGBA")
        img_data = np.array(img)
        new_img_data = np.zeros_like(img_data)
        
        # Set fully black (0, 0, 0, 255) for white pixels.
        white_pixels = (img_data[:, :, 0] == 255) & (img_data[:, :, 1] == 255) & (img_data[:, :, 2] == 255)
        new_img_data[white_pixels] = [0, 0, 0, 255]
        
        # Set fully transparent white (255, 255, 255, 0) for non-white pixels.
        new_img_data[~white_pixels] = [255, 255, 255, 0]
        
        # Convert the NumPy array back to an image and save as PNG into the buffer.
        new_img = Image.fromarray(new_img_data, "RGBA")
        new_img.save(output_buffer, format="PNG")
        return True, output_buffer.getvalue()
    except Exception as e:
        return False, str(e)

#-------------------------------------------------------------------
# Function for Interactive Correction Tab (PNG Modification)
#-------------------------------------------------------------------
def process_interactive_image(png_file):
    try:
        image = Image.open(png_file)
        image = image.convert("RGBA")
        datas = image.getdata()
        new_data = []
        for item in datas:
            # Change all white or near-white pixels to transparent
            if item[0] > 200 and item[1] > 200 and item[2] > 200:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        image.putdata(new_data)
        output_buffer = io.BytesIO()
        image.save(output_buffer, format="PNG")
        return image, output_buffer.getvalue()
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

#-------------------------------------------------------------------
# App Layout: Define Three Tabs
#-------------------------------------------------------------------
tabs = st.tabs(["SAM Plus", "Manual Seg", "Interactive Correction"])

#===================================================================
# Tab 1: SAM Plus / VNC-Dist
#===================================================================
with tabs[0]:
    st.title("🤖 1. SAM Plus / VNC-Dist")
    
    st.sidebar.header("Model Configuration")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sam_model_dir = os.path.join(base_dir, "segment-anything-main")
    checkpoint_path = os.path.join(base_dir, "sam_vit_h_4b8939.pth")
    
    model_type = st.sidebar.selectbox(
        "Model Type",
        options=["vit_b", "vit_l", "vit_h"],
        index=2,
        help="Select the SAM type.",
    )
    device = st.sidebar.selectbox(
        "Device",
        options=["cuda", "cpu"],
        index=0,
        help="Select the device to run the model on.",
    )
    st.sidebar.markdown("---")
    st.sidebar.header("Upload TIFF/TIF Images")
    uploaded_files = st.sidebar.file_uploader(
        "Choose TIFF/TIF files", type=["tiff", "tif"], accept_multiple_files=True
    )
    
    if st.sidebar.button("Process Images"):
        if not uploaded_files:
            st.error("Please upload at least one `.tiff` or `.tif` file.")
        else:
            with st.spinner("Loading SAM Plus..."):
                mask_generator = load_sam_model(checkpoint_path, model_type, device)
            # Local list to store processed images (original file name, mask filename, mask image)
            mask_image_paths = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each uploaded image.
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})")
                image = Image.open(uploaded_file).convert("RGB")
                image_np = np.array(image)
                try:
                    masks = process_image(image_np, mask_generator)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    continue
                # Save each segmented mask image.
                if len(masks) > 1:
                    for i, mask in enumerate(masks):
                        mask_pil = Image.fromarray(mask)
                        mask_filename = f"{os.path.splitext(uploaded_file.name)[0]}_{i+1}.png"
                        mask_image_paths.append((uploaded_file.name, mask_filename, mask_pil))
                else:
                    mask_pil = Image.fromarray(masks[0])
                    mask_filename = f"{os.path.splitext(uploaded_file.name)[0]}.png"
                    mask_image_paths.append((uploaded_file.name, mask_filename, mask_pil))
                
                if device == "cuda":
                    torch.cuda.empty_cache()
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.success("Processing completed!")
            progress_bar.empty()
            # Save the processed image list in session state so that it persists
            st.session_state["mask_image_paths"] = mask_image_paths

    # Preview and removal selection:
    if "mask_image_paths" in st.session_state and st.session_state["mask_image_paths"]:
        scale_factor = 0.25
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            try:
                resample_filter = Image.LANCZOS
            except AttributeError:
                resample_filter = Image.ANTIALIAS
        st.header("🔍 Segmented Worms:")
        cols = st.columns(4)
        for idx, (original_file, mask_file, mask_image) in enumerate(st.session_state["mask_image_paths"]):
            original_width, original_height = mask_image.size
            new_size = (
                max(1, int(original_width * scale_factor)),
                max(1, int(original_height * scale_factor)),
            )
            resized_mask_image = mask_image.resize(new_size, resample=resample_filter)
            col_idx = idx % 4
            with cols[col_idx]:
                st.image(resized_mask_image, caption=mask_file)
                # The checkbox marks the image for removal but keeps the preview visible.
                st.checkbox("Remove from ZIP", key=f"remove_{mask_file}")
                
        st.markdown("---")
        st.header("💾 Save Remaining Images as ZIP")
        if st.button("Generate Download for Remaining Images"):
            remaining_images = []
            # Compile regular expression to remove unwanted suffixes.
            suffix_pattern = re.compile(r'(_\d+(_corrected)?|_corrected)(?=\.png$)', re.IGNORECASE)
            for (original_file, mask_file, mask_image) in st.session_state["mask_image_paths"]:
                if not st.session_state.get(f"remove_{mask_file}", False):
                    # Rename by removing unwanted suffixes.
                    new_mask_file = suffix_pattern.sub("", mask_file)
                    remaining_images.append((new_mask_file, mask_image))
            if not remaining_images:
                st.error("All images are marked for removal. Please uncheck one or more images to keep.")
            else:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for mask_file, mask_image in remaining_images:
                        buff = io.BytesIO()
                        mask_image.save(buff, format="PNG")
                        zipf.writestr(mask_file, buff.getvalue())
                zip_buffer.seek(0)
                st.download_button(
                    label="Download Remaining Images ZIP",
                    data=zip_buffer,
                    file_name="remaining_segmented_worms.zip",
                    mime="application/zip",
                )
    
    # Sidebar Logos and Footer Text for SAM Plus Tab
    logo_path = os.path.join(base_dir, "assets", "TOH.png")
    toh_logo_path = os.path.join(base_dir, "assets", "uOttawaMed.png")
    if os.path.exists(logo_path) and os.path.exists(toh_logo_path):
        display_side_by_side_logos(logo1_path=logo_path, logo2_path=toh_logo_path, scale_factor1=0.4, scale_factor2=0.4)
    else:
        if not os.path.exists(logo_path):
            st.sidebar.warning("First logo image `uOttawaMed` not found in `assets` folder.")
        if not os.path.exists(toh_logo_path):
            st.sidebar.warning("Second logo image `TOH` not found in `assets` folder.")
    
    display_centered_text("Colavita & Perkins Lab")
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; font-style: italic; font-size: 20px; color:gray;">
            "Colavita & Perkins Lab"
        </div>
        """,
        unsafe_allow_html=True,
    )

#===================================================================
# Tab 2: Manual Seg (TIFF to PNG Conversion with Preview and ZIP Download)
#===================================================================
with tabs[1]:
    st.header("Manual Segmentation")
    st.markdown("Select one or more TIFF images (.tif or .tiff) below:")
    
    manual_images = st.file_uploader(
        "Choose TIFF/TIF images", type=["tiff", "tif"], accept_multiple_files=True
    )
    
    if st.button("Process Selected Images", key="manual_seg"):
        if not manual_images:
            st.error("Please select at least one TIFF image.")
        else:
            # Dictionary to store processed image data (PNG bytes) keyed by file name.
            processed_data = {}
            errors = []
            
            # Process each uploaded image.
            for tiff_file in manual_images:
                base_name = os.path.splitext(tiff_file.name)[0]
                # Create an in-memory buffer to save the PNG.
                output_buffer = io.BytesIO()
                success, result = convert_tiff_to_png(tiff_file, output_buffer)
                if success:
                    processed_data[f"{base_name}.png"] = result
                else:
                    errors.append(f"{tiff_file.name}: {result}")
            
            if processed_data:
                st.success(f"Processed {len(processed_data)} image(s) successfully!")
                st.markdown("**Preview of Processed Images:**")
                
                # Preview images in a grid of 4 columns.
                cols = st.columns(4)
                for idx, (filename, png_bytes) in enumerate(processed_data.items()):
                    image = Image.open(io.BytesIO(png_bytes))
                    col = cols[idx % 4]
                    with col:
                        st.image(image, caption=filename)
                
                # Create an in-memory ZIP file with the processed images.
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for filename, png_bytes in processed_data.items():
                        zipf.writestr(filename, png_bytes)
                zip_buffer.seek(0)
                
                st.download_button(
                    label="Download Processed Images ZIP",
                    data=zip_buffer,
                    file_name="processed_images(manual).zip",
                    mime="application/zip",
                )
            if errors:
                st.error("Some errors occurred:")
                for err in errors:
                    st.error(err)

#===================================================================
# Tab 3: Interactive Correction
#===================================================================
with tabs[2]:
    st.header("Interactive Correction")
    st.markdown("Upload your modified PNG images for further correction:")
    
    png_files = st.file_uploader("Choose PNG images", type=["png"], accept_multiple_files=True)
    
    if st.button("Process Images", key="interactive_corr"):
        if not png_files:
            st.error("Please upload at least one PNG image.")
        else:
            processed_data = {}
            preview_images = []
            
            for png_file in png_files:
                processed_image, processed_bytes = process_interactive_image(png_file)
                if processed_image is not None:
                    base_name = os.path.splitext(png_file.name)[0]
                    new_filename = f"{base_name}_corrected.png"
                    processed_data[new_filename] = processed_bytes
                    preview_images.append((new_filename, processed_image))
            
            if processed_data:
                st.success(f"Processed {len(processed_data)} image(s) successfully!")
                st.markdown("**Preview of Corrected Images:**")
                
                # Display previews in a grid of 4 columns.
                cols = st.columns(4)
                for idx, (filename, image) in enumerate(preview_images):
                    col = cols[idx % 4]
                    with col:
                        st.image(image, caption=filename)
                
                # Automatically apply suffix correction before saving as ZIP.
                suffix_pattern = re.compile(r'(_\d+(_corrected)?|_corrected)(?=\.png$)', re.IGNORECASE)
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for filename, image_bytes in processed_data.items():
                        new_filename = suffix_pattern.sub("", filename)
                        zipf.writestr(new_filename, image_bytes)
                zip_buffer.seek(0)
                
                st.download_button(
                    label="Download Corrected Images ZIP",
                    data=zip_buffer,
                    file_name="corrected_images.zip",
                    mime="application/zip",
                )
