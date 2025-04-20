#VNC-Quant/VNC-Dist    Colavita & Perkins Lab, Faculty of Medicine UOttawa
import streamlit as st
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import threading
import time
import os
import tempfile
import zipfile
import io
from pathlib import Path
from skimage.morphology import skeletonize, dilation, square
from scipy.ndimage import convolve
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
from PIL import Image
import numpy as np
import cv2
import warnings
import base64
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from skimage.graph import route_through_array

warnings.filterwarnings("ignore")

try:
    from PIL import Image
    resample_method = Image.Resampling.LANCZOS
except AttributeError:
    resample_method = Image.LANCZOS

TIMEOUT_SECONDS = 300

def inactivity_monitor():
    while True:
        time.sleep(10)
        if time.time() - st.session_state["last_interaction"] > TIMEOUT_SECONDS:
            st.write(f"No activity detected for {TIMEOUT_SECONDS} seconds. Shutting down.")
            sys.exit(0)

if "last_interaction" not in st.session_state:
    st.session_state["last_interaction"] = time.time()

if "inactivity_thread" not in st.session_state:
    t = threading.Thread(target=inactivity_monitor, daemon=True)
    st.session_state["inactivity_thread"] = t
    t.start()

st.session_state["last_interaction"] = time.time()

neurons_of_interest = [
    "SABVR", "SABD", "DA1", "DA2", "DA3", "DA4", "DA5", "DA6", "DA7",
    "DA8", "DA9", "DD1", "DD2", "DD3", "DD4", "DD5", "DD6", "DB1", "DB2",
    "DB3", "DB4", "DB5", "DB6", "DB7",
]

def qq_shapiro_analysis(data_files, genotype_names):
    datasets = []
    for file_obj, genotype in zip(data_files, genotype_names):
        try:
            file_obj.seek(0)
            df = pd.read_csv(file_obj)
            required_cols = ["filename"] + neurons_of_interest
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                st.error(f"File {file_obj.name} is missing columns: {missing}")
                continue
            df = df[required_cols]
            df["Genotype"] = genotype
            datasets.append(df)
        except Exception as e:
            st.error(f"Error reading file {file_obj.name}: {e}")
    if not datasets:
        st.error("No valid datasets loaded.")
        return None, None

    combined_data = pd.concat(datasets, ignore_index=True)
    genotypes_ordered = genotype_names
    n_genotypes = len(genotypes_ordered)
    cmap = plt.cm.get_cmap("tab10", n_genotypes)
    genotype_colors = {genotype: cmap(i) for i, genotype in enumerate(genotypes_ordered)}

    shapiro_results = []
    n_neurons = len(neurons_of_interest)
    n_cols = 3
    n_rows = math.ceil(n_neurons / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for i, neuron in enumerate(neurons_of_interest):
        ax = axes[i]
        handles = []
        labels = []
        for genotype in genotypes_ordered:
            neuron_data = combined_data[combined_data["Genotype"] == genotype][neuron].dropna()
            if len(neuron_data) > 3:
                shapiro_stat, shapiro_pval = stats.shapiro(neuron_data)
                shapiro_results.append({
                    "Neuron": neuron,
                    "Genotype": genotype,
                    "W": shapiro_stat,
                    "P_Value": shapiro_pval,
                    "Normal": "Yes" if shapiro_pval > 0.05 else "No",
                })
                (osm, osr), (slope, intercept, r) = stats.probplot(neuron_data, dist="norm")
                sc = ax.scatter(osm, osr, label=genotype, color=genotype_colors[genotype], s=10)
                handles.append(sc)
                labels.append(genotype)
                ax.plot(osm, slope * osm + intercept, color=genotype_colors[genotype], linestyle="--")
        if neuron.startswith("DA"):
            title_color = "darkgreen"
        elif neuron.startswith("DD"):
            title_color = "darkred"
        elif neuron.startswith("DB"):
            title_color = "orange"
        else:
            title_color = "black"
        ax.set_title(neuron, fontsize=11, weight="bold", color=title_color)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        if neuron == "SABVR" and handles:
            ax.legend(handles=handles, labels=labels, title="", fontsize="small", title_fontsize="small")
    for j in range(n_neurons, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    shapiro_df = pd.DataFrame(shapiro_results).sort_values(by=["Neuron", "Genotype"])
    return fig, shapiro_df

def violin_box_plot(data_files, genotype_names):
    datasets = []
    for file_obj, genotype in zip(data_files, genotype_names):
        try:
            file_obj.seek(0)
            data = pd.read_csv(file_obj)
            data["Genotype"] = genotype
            datasets.append(data)
        except Exception as e:
            st.error(f"Error reading a file: {e}")
    if not datasets:
        st.error("No datasets loaded.")
        return None

    combined_data = pd.concat(datasets, ignore_index=True)
    neurons_of_interest_violin = [
        "DA1", "DA2", "DA3", "DA4", "DA5", "DA6", "DA7", "DA8", "DA9",
        "DD1", "DD2", "DD3", "DD4", "DD5", "DD6",
        "DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7",
    ]
    genotypes_ordered = genotype_names
    base_colors = ["gray", "red", "cyan", "green", "purple", "orange", "blue", "magenta"]
    genotype_colors = {gt: base_colors[i % len(base_colors)] for i, gt in enumerate(genotypes_ordered)}
    title_colors = {"DA": "darkgreen", "DD": "darkred", "DB": "darkorange"}
    num_neurons = len(neurons_of_interest_violin)
    num_cols = 3
    num_rows = math.ceil(num_neurons / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))
    axes = axes.flatten()
    positions = list(range(len(genotypes_ordered), 0, -1))

    for idx, neuron in enumerate(neurons_of_interest_violin):
        ax = axes[idx]
        data_list = [combined_data[combined_data["Genotype"] == genotype][neuron].dropna() for genotype in genotypes_ordered]
        vp = ax.violinplot(data_list, positions=positions, vert=False, showmeans=True, showextrema=False, showmedians=False, widths=0.8)
        for i, body in enumerate(vp["bodies"]):
            color = genotype_colors[genotypes_ordered[i]]
            body.set_facecolor(color)
            body.set_alpha(0.4)
            body.set_edgecolor("none")
        if "means" in vp:
            for mean_line in vp["means"]:
                mean_line.set_marker("D")
                mean_line.set_markersize(6)
                mean_line.set_markerfacecolor("white")
                mean_line.set_markeredgecolor("black")
        bp = ax.boxplot(data_list, patch_artist=True, vert=False, labels=genotypes_ordered, positions=positions, widths=0.3)
        for patch, genotype in zip(bp["boxes"], genotypes_ordered):
            patch.set_facecolor(genotype_colors[genotype])
            patch.set_alpha(0.7)
            patch.set_linewidth(1)
        for median in bp["medians"]:
            median.set(color="black", linewidth=2)
        for pos, d, genotype in zip(positions, data_list, genotypes_ordered):
            jitter = np.random.normal(0, 0.04, size=len(d))
            ax.scatter(d, np.full_like(d, pos) + jitter, color=genotype_colors[genotype], alpha=0.7, s=10, edgecolors="k")
        neuron_type = neuron[:2]
        title_color = title_colors.get(neuron_type, "black")
        ax.set_title(neuron, color=title_color, fontsize=12, weight="bold")
        ax.set_xlabel("Relative position%")
        ax.set_ylabel("")
        ax.tick_params(axis="y", rotation=0)
    for i in range(num_neurons, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    return fig

def process_csv(input_path):
    data = pd.read_csv(input_path)
    columns_to_keep = [
        "DA", "DD", "DB",
        "Cumulative Length DA", "Normalized Cumulative Length DA",
        "Cumulative Length DD", "Normalized Cumulative Length DD",
        "Cumulative Length DB", "Normalized Cumulative Length DB",
    ]
    filtered_data = data[[col for col in columns_to_keep if col in data.columns]]
    observations = {
        "DA": ["SABVL", "SABVR", "SABD", "DA1", "DA2", "DA3", "DA4", "DA5", "DA6", "DA7", "DA8", "DA9", "Rectum"],
        "DD": ["DD1", "DD2", "DD3", "DD4", "DD5", "DD6"],
        "DB": ["DB1", "DB2", "DB3", "DB4", "DB5", "DB6", "DB7"],
    }
    pivot_data = pd.DataFrame()
    for key, obs_list in observations.items():
        if f"Normalized Cumulative Length {key}" not in filtered_data.columns:
            continue
        for obs in obs_list:
            col_name = f"Normalized Cumulative Length {key}"
            subset = filtered_data[filtered_data[key] == obs]
            if not subset.empty:
                filtered_obs_data = subset[col_name].reset_index(drop=True)
                pivot_data[obs] = filtered_obs_data
            else:
                pivot_data[obs] = pd.Series(dtype=float)
    return pivot_data

def find_endpoints(skeleton):
    endpoints = []
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x]:
                neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - 1
                if neighbors == 1:
                    endpoints.append((x, y))
    return endpoints

class WormMeasure:
    def __init__(self, image_file, csv_file, projection_images_dir, relative_distances_dir):
        self.image_file = image_file
        self.csv_file = csv_file
        self.projection_images_dir = projection_images_dir
        self.relative_distances_dir = relative_distances_dir
        self.process_files()

    def process_files(self):
        self.edited_df = self.edit_csv(self.csv_file)
        self.load_image()
        image_base_name = os.path.splitext(os.path.basename(self.image_file.name))[0]
        self.output_image_path = os.path.join(self.projection_images_dir, f"{image_base_name}_projection.png")
        self.output_csv_path = os.path.join(self.relative_distances_dir, f"{image_base_name}_RD.csv")
        if self.output_image_path and self.output_csv_path:
            self.create_black_background_image()
            self.process_image()
            self.analyze_spline()

    def load_image(self):
        self.color_worm_img = Image.open(self.image_file)
        self.color_worm_arr = np.array(self.color_worm_img)

    def create_black_background_image(self):
        if self.color_worm_arr.shape[2] == 4:
            alpha_channel = self.color_worm_arr[:, :, 3]
        else:
            alpha_channel = 255 * np.ones(self.color_worm_arr.shape[:2], dtype=self.color_worm_arr.dtype)
        black_background_arr = np.zeros(self.color_worm_arr.shape[:2] + (3,), dtype=self.color_worm_arr.dtype)
        worm_mask = alpha_channel != 0
        black_background_arr[worm_mask, :3] = 255
        black_background_arr = black_background_arr.astype(np.uint8)
        if self.color_worm_arr.shape[2] == 4:
            self.image = cv2.cvtColor(black_background_arr, cv2.COLOR_RGBA2BGR)
        else:
            self.image = cv2.cvtColor(black_background_arr, cv2.COLOR_RGB2BGR)

    def process_image(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        skeleton = skeletonize(binary_image == 255)
        self.skeleton = skeleton
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contour_image = self.image.copy()
        cv2.drawContours(self.contour_image, contours, -1, (0, 255, 0), thickness=-1)
        self.overlay_image = self.contour_image.copy()

    def fit_spline_to_contour(self, contour):
        t = np.linspace(0, 1, len(contour))
        spline_x = UnivariateSpline(t, contour[:, 0], k=3, s=900)
        spline_y = UnivariateSpline(t, contour[:, 1], k=3, s=900)
        x_fit = spline_x(t)
        y_fit = spline_y(t)
        return x_fit, y_fit, t, spline_x, spline_y

    def project_point_onto_spline(self, point, spline_x, spline_y, t):
        distances = (spline_x(t) - point[0]) ** 2 + (spline_y(t) - point[1]) ** 2
        t_closest_index = np.argmin(distances)
        t_closest = t[t_closest_index]
        x_closest = spline_x(t_closest)
        y_closest = spline_y(t_closest)
        return x_closest, y_closest, t_closest

    def reorder_contour_based_on_da1(self, contour_points, da1):
        distances = np.linalg.norm(contour_points - da1, axis=1)
        start_index = np.argmin(distances)
        offset = int(len(contour_points) / 2.24)
        adjusted_start_index = (start_index - offset) % len(contour_points)
        return np.roll(contour_points, -adjusted_start_index, axis=0)

    def analyze_spline(self):
        if self.edited_df is not None:
            df = self.edited_df.copy()
            df["DA"] = df["DA"].fillna("Unknown")
            df["DD"] = df["DD"].fillna("Unknown")
            df["DB"] = df["DB"].fillna("Unknown")
        gray_image = cv2.cvtColor(self.overlay_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv2.contourArea)
        contour_points = np.squeeze(largest_contour)
        da1 = df[df["DA"] != "Unknown"][["X", "Y"]].values[0]
        reordered_contour_points = self.reorder_contour_based_on_da1(contour_points, da1)
        x_fit, y_fit, t, spline_x, spline_y = self.fit_spline_to_contour(reordered_contour_points)
        projected_points_da = np.array([
            self.project_point_onto_spline(point, spline_x, spline_y, t)
            for point in df[df["DA"] != "Unknown"][["X", "Y"]].values
        ])
        projected_points_dd = np.array([
            self.project_point_onto_spline(point, spline_x, spline_y, t)
            for point in df[df["DD"] != "Unknown"][["X", "Y"]].values
        ])
        projected_points_db = np.array([
            self.project_point_onto_spline(point, spline_x, spline_y, t)
            for point in df[df["DB"] != "Unknown"][["X", "Y"]].values
        ])
        da_points = df[df["DA"] != "Unknown"][["X", "Y"]].values
        print("Projected Points DA:")
        for original_point, projection_result in zip(da_points, projected_points_da):
            print(f"Original: {original_point}, Projected: ({projection_result[0]:.3f}, {projection_result[1]:.3f}), t: {projection_result[2]:.3f}")
        dd_points = df[df["DD"] != "Unknown"][["X", "Y"]].values
        print("\nProjected Points DD:")
        for original_point, projection_result in zip(dd_points, projected_points_dd):
            print(f"Original: {original_point}, Projected: ({projection_result[0]:.3f}, {projection_result[1]:.3f}), t: {projection_result[2]:.3f}")
        db_points = df[df["DB"] != "Unknown"][["X", "Y"]].values
        print("\nProjected Points DB:")
        for original_point, projection_result in zip(db_points, projected_points_db):
            print(f"Original: {original_point}, Projected: ({projection_result[0]:.3f}, {projection_result[1]:.3f}), t: {projection_result[2]:.3f}")

        def Integrand(t_val, spline_x_func, spline_y_func):
            dx_dt = spline_x_func.derivative()(t_val)
            dy_dt = spline_y_func.derivative()(t_val)
            return np.sqrt(dx_dt**2 + dy_dt**2)

        arc_lengths_da = []
        for i in range(len(projected_points_da) - 1):
            t1 = projected_points_da[i][-1]
            t2 = projected_points_da[i + 1][-1]
            segment_length, _ = quad(Integrand, t1, t2, args=(spline_x, spline_y), epsabs=1e-6, limit=100)
            arc_lengths_da.append(abs(segment_length))
        da13_idx = df[df["DA"] == "DA13"].index[0]
        cumulative_lengths_da = np.cumsum([0] + arc_lengths_da).tolist()
        if da13_idx < len(cumulative_lengths_da):
            max_cumulative_length_da = cumulative_lengths_da[da13_idx]
            normalized_cumulative_lengths_da = [round(100.0 * length / max_cumulative_length_da, 2) for length in cumulative_lengths_da]
        else:
            print("Error: DA13 index is out of range.")
            normalized_cumulative_lengths_da = [0] * len(cumulative_lengths_da)
        arc_lengths_dd = []
        for i in range(len(projected_points_dd) - 1):
            t1 = projected_points_dd[i][-1]
            t2 = projected_points_dd[i + 1][-1]
            segment_length, _ = quad(Integrand, t1, t2, args=(spline_x, spline_y), epsabs=1e-6, limit=100)
            arc_lengths_dd.append(abs(segment_length))
        da1_idx = 0
        dd1_idx = 0
        t_da1 = projected_points_da[da1_idx][-1]
        t_dd1 = projected_points_dd[dd1_idx][-1]
        curve_length_da1_to_dd1, _ = quad(Integrand, t_da1, t_dd1, args=(spline_x, spline_y), epsabs=1e-6, limit=100)
        curve_length_da1_to_dd1 = abs(curve_length_da1_to_dd1)
        cumulative_lengths_dd = np.cumsum([curve_length_da1_to_dd1] + arc_lengths_dd).tolist()
        normalized_cumulative_lengths_dd = [round(100.0 * length / cumulative_lengths_da[da13_idx], 2) for length in cumulative_lengths_dd]
        arc_lengths_db = []
        for i in range(len(projected_points_db) - 1):
            t1 = projected_points_db[i][-1]
            t2 = projected_points_db[i + 1][-1]
            segment_length, _ = quad(Integrand, t1, t2, args=(spline_x, spline_y), epsabs=1e-6, limit=100)
            arc_lengths_db.append(abs(segment_length))
        t_da1 = projected_points_da[0][-1]
        t_da13 = projected_points_da[da13_idx][-1] if len(projected_points_da) > da13_idx else t_da1
        t_db1 = projected_points_db[0][-1]
        curve_length_da1_to_db1, _ = quad(Integrand, t_da1, t_db1, args=(spline_x, spline_y), epsabs=1e-6, limit=100)
        if not (t_da1 < t_db1 < t_da13) and not (t_da1 > t_db1 > t_da13):
            curve_length_da1_to_db1 = -abs(curve_length_da1_to_db1)
        else:
            curve_length_da1_to_db1 = abs(curve_length_da1_to_db1)
        cumulative_lengths_db = np.cumsum([curve_length_da1_to_db1] + arc_lengths_db).tolist()
        normalized_cumulative_lengths_db = [round(100.0 * length / cumulative_lengths_da[da13_idx], 2) for length in cumulative_lengths_db]

        da_renaming_map = {
            "DA1": "SABVL", "DA2": "SABVR", "DA3": "SABD", "DA4": "DA1", "DA5": "DA2",
            "DA6": "DA3", "DA7": "DA4", "DA8": "DA5", "DA9": "DA6", "DA10": "DA7",
            "DA11": "DA8", "DA12": "DA9", "DA13": "Rectum",
        }
        df["DA"] = df["DA"].replace(da_renaming_map)
        df.loc[df["DA"] != "Unknown", "Cumulative Length DA"] = cumulative_lengths_da
        df.loc[df["DA"] != "Unknown", "Normalized Cumulative Length DA"] = normalized_cumulative_lengths_da
        df.loc[df["DD"] != "Unknown", "Cumulative Length DD"] = cumulative_lengths_dd
        df.loc[df["DD"] != "Unknown", "Normalized Cumulative Length DD"] = normalized_cumulative_lengths_dd
        df.loc[df["DB"] != "Unknown", "Cumulative Length DB"] = cumulative_lengths_db
        df.loc[df["DB"] != "Unknown", "Normalized Cumulative Length DB"] = normalized_cumulative_lengths_db

        image_name = os.path.splitext(os.path.basename(self.image_file.name))[0]
        print("Shape of self.color_worm_arr:", self.color_worm_arr.shape)
        print(f"Length of the largest contour: {len(contour_points)}")
        plt.figure(figsize=(10, 10))
        image_with_lightgray_background = self.image.copy()
        image_with_lightgray_background[binary_image == 0] = [211, 211, 211]
        plt.imshow(cv2.cvtColor(image_with_lightgray_background, cv2.COLOR_BGR2RGB))
        plt.plot(x_fit, y_fit, color="blue", linewidth=2, label="Spline for Contour")
        plt.scatter(df["X"], df["Y"], c="black", marker="*", s=60, label="All Neurons (25)")
        is_label_added_da = False
        df_da = df[df["DA"] != "Unknown"][["X", "Y", "DA"]]
        projected_points_da = np.array([
            self.project_point_onto_spline(point, spline_x, spline_y, t)
            for point in df_da[["X", "Y"]].values
        ])
        for index, (point, row) in enumerate(zip(projected_points_da, df_da.itertuples())):
            x_closest, y_closest, t_value = point
            label = row.DA
            color = "skyblue" if label in ["SABVL", "Rectum"] else "green"
            print(f"Index: {index}, Label: {label}, Projected Point: ({x_closest}, {y_closest})")
            if not is_label_added_da:
                plt.scatter(x_closest, y_closest, color=color, marker="v", s=70, label="Projected DA")
                is_label_added_da = True
            else:
                plt.scatter(x_closest, y_closest, color=color, marker="v", s=70)
            plt.annotate(f"{t_value:.2f}", (x_closest, y_closest), textcoords="offset points", xytext=(0, 10), ha="center", color=color)
        is_label_added_dd = False
        for point in projected_points_dd:
            x_closest, y_closest, t_value = point
            if not is_label_added_dd:
                plt.scatter(x_closest, y_closest, color="red", marker="o", s=65, label="Projected DD")
                is_label_added_dd = True
            else:
                plt.scatter(x_closest, y_closest, color="red", marker="o", s=65)
            plt.annotate(f"{t_value:.2f}", (x_closest, y_closest), textcoords="offset points", xytext=(0, 10), ha="center", color="red")
        is_label_added_db = False
        for point in projected_points_db:
            x_closest, y_closest, t_value = point
            if not is_label_added_db:
                plt.scatter(x_closest, y_closest, color="darkgoldenrod", marker="o", s=60, label="Projected DB")
                is_label_added_db = True
            else:
                plt.scatter(x_closest, y_closest, color="darkgoldenrod", marker="o", s=60)
            plt.annotate(f"{t_value:.2f}", (x_closest, y_closest), textcoords="offset points", xytext=(0, 10), ha="center", color="darkgoldenrod")
        for index, row in df.iterrows():
            if not pd.isna(row["DA"]) and row["DA"] != "Unknown":
                plt.text(row["X"], row["Y"], str(row["DA"]), color="green", fontsize=1)
            elif not pd.isna(row["DD"]) and row["DD"] != "Unknown":
                plt.text(row["X"], row["Y"], str(row["DD"]), color="red", fontsize=1)
            elif not pd.isna(row["DB"]) and row["DB"] != "Unknown":
                plt.text(row["X"], row["Y"], str(row["DB"]), color="darkgoldenrod", fontsize=1)
        if hasattr(self, 'skeleton'):
            endpoints = find_endpoints(self.skeleton)
            if len(endpoints) >= 2:
                max_dist = 0
                best_pair = None
                for i in range(len(endpoints)):
                    for j in range(i+1, len(endpoints)):
                        p1 = endpoints[i]
                        p2 = endpoints[j]
                        dist = np.linalg.norm(np.array(p1) - np.array(p2))
                        if dist > max_dist:
                            max_dist = dist
                            best_pair = (p1, p2)
                if best_pair:
                    start = (best_pair[0][1], best_pair[0][0])  # (y, x)
                    end = (best_pair[1][1], best_pair[1][0])  # (y, x)
                    cost_array = np.where(self.skeleton, 1, 10000).astype(float)
                    path, _ = route_through_array(cost_array, start=start, end=end, fully_connected=True)
                    path_x = [p[1] for p in path]  # column
                    path_y = [p[0] for p in path]  # row
                    plt.plot(path_x, path_y, color="black", linestyle="dashed", linewidth=1, label="Midline")
        plt.text(self.image.shape[1] / 2, -4, image_name, color="black", fontsize=10, backgroundcolor="lightblue")
        plt.axis("off")
        plt.savefig(self.output_image_path, bbox_inches="tight", pad_inches=1)
        plt.close()
        df.to_csv(self.output_csv_path, index=False)

    def edit_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        df = df.drop(df.columns[0], axis=1, errors="ignore")
        df = df.drop("Count", axis=1, errors="ignore")
        df = df.drop("Slice", axis=1, errors="ignore")
        df = df.drop("Ch", axis=1, errors="ignore")
        df["X"] = (df["X"] * 6.8259).round(3)
        df["Y"] = (df["Y"] * 6.8259).round(3)
        df["DA"] = (df["Counter"] == 1).astype(int)
        df["DD"] = (df["Counter"] == 2).astype(int)
        df["DB"] = (df["Counter"] == 3).astype(int)
        df = df.drop("Counter", axis=1)
        counter_da = 0
        counter_dd = 0
        counter_db = 0
        for idx, row in df.iterrows():
            if row["DA"] == 1:
                counter_da += 1
                df.at[idx, "DA"] = f"DA{counter_da}"
            elif row["DD"] == 1:
                counter_dd += 1
                df.at[idx, "DD"] = f"DD{counter_dd}"
            elif row["DB"] == 1:
                counter_db += 1
                df.at[idx, "DB"] = f"DB{counter_db}"
        df["DA"] = df["DA"].astype(object)
        df["DD"] = df["DD"].astype(object)
        df["DB"] = df["DB"].astype(object)
        df["DA"].replace(0, np.nan, inplace=True)
        df["DD"].replace(0, np.nan, inplace=True)
        df["DB"].replace(0, np.nan, inplace=True)
        return df

def determine_color(feature_name, is_pry1, is_background=False):
    if is_background:
        return "rgba(128, 128, 128, 0)"
    match = re.match(r"([A-Z]+)(\d+)", feature_name)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
    else:
        prefix = feature_name
        number = 1
    if prefix.startswith("DD"):
        return "#E31A1C" if number % 2 == 1 else "#FF7F7F"
    elif prefix.startswith("DA"):
        return "#33a02c" if number % 2 == 1 else "#90EE90"
    elif prefix.startswith("DB"):
        return "#E69F00" if number % 2 == 1 else "#FFB84D"
    else:
        return "rgba(192,192,192,1)"

def determine_y_values(feature, fixed_y_value, fixed_y_value_db, fixed_y_value_da8):
    if feature in ["DB1", "DB5", "DB6"]:
        return fixed_y_value - 3
    elif feature in ["DD1", "DD3", "DD5", "DA2", "DA4", "DA6", "DA9"]:
        return fixed_y_value + 6
    else:
        return fixed_y_value

def calculate_confidence_interval(data, confidence=0.95):
    mean = data.mean()
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, ci

def calculate_weighted_mean(values, weights):
    return np.average(values, weights=weights)

def prepare_traces(df_melted, fixed_y_value, fixed_y_value_db, fixed_y_value_da8, is_pry1, is_background=False):
    traces = []
    means = {}
    y_values = {}
    jitter_strength = 0.6
    t_line_width = 2.5
    for feature, group_data in df_melted.groupby("Feature"):
        color = determine_color(feature, is_pry1, is_background)
        y_value = determine_y_values(feature, fixed_y_value, fixed_y_value_db, fixed_y_value_da8)
        values = group_data["Value"].dropna()
        if len(values) == 0:
            continue
        value_counts = values.value_counts()
        weights = values.map(value_counts)
        mean_value = calculate_weighted_mean(values, weights)
        mean_t, ci_t = calculate_confidence_interval(values)
        y_values_jittered = np.random.normal(y_value, jitter_strength, len(group_data))
        scatter_trace = go.Scatter(
            y=y_values_jittered,
            x=group_data["Value"],
            mode="markers",
            marker=dict(color=color, size=8),
            name=feature,
            showlegend=False,
        )
        traces.append(scatter_trace)
        if not is_background:
            ci_y_value = y_value + 4 if is_pry1 else y_value - 4
            ci_trace = go.Scatter(
                x=[mean_t - ci_t, mean_t + ci_t],
                y=[ci_y_value, ci_y_value],
                mode="lines",
                line=dict(color=color, width=t_line_width),
                showlegend=False,
                name=f"{feature} CI",
            )
            traces.append(ci_trace)
            mean_trace = go.Scatter(
                x=[mean_t, mean_t],
                y=[y_value, ci_y_value],
                mode="lines",
                line=dict(color=color, width=t_line_width),
                showlegend=False,
                name=f"{feature} mean",
            )
            traces.append(mean_trace)
        means[feature] = mean_value
        y_values[feature] = y_value
    return traces, means, y_values

def create_subplots(traces, fixed_y_value, fixed_y_value_db, fixed_y_value_da8, plot_bgcolor):
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0, shared_xaxes=True)
    for trace in traces:
        if "DD" in trace.name:
            fig.add_trace(trace, row=1, col=1)
        elif "DB" in trace.name:
            fig.add_trace(trace, row=2, col=1)
        elif "DA" in trace.name:
            fig.add_trace(trace, row=3, col=1)
        else:
            fig.add_trace(trace, row=3, col=1)
    reference_values = [0, 100]
    for reference_value in reference_values:
        for row in range(1, 4):
            fig.add_shape(
                type="line",
                x0=reference_value,
                x1=reference_value,
                y0=0,
                y1=1,
                xref=f"x{row}",
                yref="paper",
                line=dict(color="black", width=1.5, dash="solid"),
            )
    all_values = [trace.x for trace in traces if isinstance(trace, go.Scatter)]
    all_values = [item for sublist in all_values for item in sublist]
    x_min, x_max = min(all_values), max(all_values)
    x_range = [x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min)]
    fig.update_layout(
        xaxis=dict(
            range=x_range,
            showticklabels=False,
            showgrid=False,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2,
        ),
        xaxis2=dict(
            range=x_range,
            showticklabels=False,
            showgrid=False,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2,
        ),
        xaxis3=dict(
            range=x_range,
            showticklabels=True,
            showgrid=False,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=2,
            tickmode="array",
            tickvals=[-8, -5, 0, 5, 25, 50, 75, 100],
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis2=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis3=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor="rgb(255, 255, 255)",
        showlegend=False,
        height=365,
        width=1150,
    )
    return fig

def rank_features(df):
    columns_to_rank = [col for col in df.columns if col not in ["filename", "Rectum", "SABVL", "SABVR", "SABD"]]
    df_ranked = df.copy()
    df_ranked[columns_to_rank] = df[columns_to_rank].rank(axis=1, method="min").astype(int)
    return df_ranked

def process_csv_file(input_file):
    df = pd.read_csv(input_file)
    df_ranked = rank_features(df)
    return df_ranked

def create_bar_plot_on_ax(df_ranked, plot_title, ax):
    columns_to_rank = [col for col in df_ranked.columns if col not in ["filename", "Rectum", "SABVL", "SABVR", "SABD"]]
    df_ranked_sorted = df_ranked.sort_values(by=columns_to_rank)
    for i, (index, row) in enumerate(df_ranked_sorted.iterrows()):
        ranks = row[columns_to_rank].values
        classes = [col[:2] for col in columns_to_rank]
        sorted_ranks_classes = sorted(zip(ranks, classes))
        sorted_ranks, sorted_classes = zip(*sorted_ranks_classes)
        rank_occupancy = {rank: 0 for rank in sorted_ranks}
        color_map = {
            "DA": (0/255, 128/255, 0/255),
            "DD": (255/255, 0/255, 255/255),
            "DB": (200/255, 200/255, 200/255),
        }
        colors = [color_map[cls] for cls in sorted_classes]
        for rank, color in zip(sorted_ranks, colors):
            width = 1 / (rank_occupancy[rank] + 1)
            ax.barh(
                y=i,
                width=width,
                height=0.8,
                left=rank - 1 + rank_occupancy[rank] * width,
                color=color,
            )
            rank_occupancy[rank] += 1
    ax.set_title(plot_title, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def da_calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1)*var1 + (n2 - 1)*var2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / pooled_std
    return d

def da_load_and_prepare_data(file_obj, type_label):
    df = pd.read_csv(file_obj)
    df["Type"] = type_label
    df_melted = df.melt(id_vars="Type", var_name="Feature", value_name="Value")
    df_melted["Value"] = pd.to_numeric(df_melted["Value"], errors="coerce")
    df_numeric = df.drop(df.columns[0], axis=1, errors="ignore")
    return df_melted, df_numeric

def da_perform_conditional_t_tests(df1, df2):
    results = {}
    for column in df1.columns:
        if column in df2.columns:
            col1 = pd.to_numeric(df1[column], errors="coerce").dropna()
            col2 = pd.to_numeric(df2[column], errors="coerce").dropna()
            aligned_col1, aligned_col2 = col1.align(col2, join="inner")
            if np.allclose(aligned_col1, aligned_col2, atol=1e-8):
                continue
            stat, p = stats.levene(aligned_col1, aligned_col2)
            if p > 0.05:
                t_stat, p_value = stats.ttest_ind(aligned_col1, aligned_col2, equal_var=True)
                df_val = len(aligned_col1) + len(aligned_col2) - 2
            else:
                t_stat, p_value = stats.ttest_ind(aligned_col1, aligned_col2, equal_var=False)
                s1 = np.var(aligned_col1, ddof=1)
                s2 = np.var(aligned_col2, ddof=1)
                n1 = len(aligned_col1)
                n2 = len(aligned_col2)
                df_val = (s1/n1 + s2/n2)**2 / (((s1/n1)**2/(n1-1)) + ((s2/n2)**2/(n2-1)))
            if p_value < 0.05:
                if p_value < 0.001:
                    sig = "***"
                elif p_value < 0.01:
                    sig = "**"
                else:
                    sig = "*"
                effect = da_calculate_cohens_d(aligned_col1, aligned_col2)
                results[column] = (sig, effect, t_stat, df_val)
    return results

def da_determine_color(feature_name, is_pry1, is_background=False, for_marker=True):
    DD_DARK = "#E31A1C"
    DD_LIGHT = "#FBB4AE"
    DA_DARK = "#33a02c"
    DA_LIGHT = "#b2df8a"
    DB_DARK = "#E69F00"
    DB_LIGHT = "#FFDB9C"
    WT_LIGHT = "#CCCCCC"
    WT_DARK = "#666666"
    if is_background:
        return "rgba(0,0,0,0)"
    if for_marker:
        if not is_pry1:
            m = re.search(r"\d+", feature_name)
            if m:
                num = int(m.group())
                return WT_LIGHT if num % 2 == 1 else WT_DARK
            else:
                return WT_DARK
        else:
            if feature_name.startswith("DD"):
                m = re.search(r"\d+", feature_name)
                if m:
                    num = int(m.group())
                    return DD_LIGHT if num % 2 == 1 else DD_DARK
                else:
                    return DD_DARK
            elif feature_name.startswith("DA"):
                m = re.search(r"\d+", feature_name)
                if m:
                    num = int(m.group())
                    return DA_LIGHT if num % 2 == 1 else DA_DARK
                else:
                    return DA_DARK
            elif feature_name.startswith("DB"):
                m = re.search(r"\d+", feature_name)
                if m:
                    num = int(m.group())
                    return DB_LIGHT if num % 2 == 1 else DB_DARK
                else:
                    return DB_DARK
            elif "SAB" in feature_name:
                return "cyan"
            else:
                return "black"
    else:
        if not is_pry1:
            return WT_DARK
        else:
            if feature_name.startswith("DD"):
                return DD_DARK
            elif feature_name.startswith("DA"):
                return DA_DARK
            elif feature_name.startswith("DB"):
                return DB_DARK
            elif "SAB" in feature_name:
                return "cyan"
            else:
                return "black"

def da_determine_y_values(feature, fixed_y_value, fixed_y_value_db, fixed_y_value_da8):
    if feature in ["DB1", "DB5", "DB6"]:
        return fixed_y_value - 1.2
    elif feature in ["DD1", "DD3", "DD5", "DA2", "DA4", "DA6", "DA9"]:
        return fixed_y_value + 2.5
    else:
        return fixed_y_value

def da_calculate_confidence_interval(data, confidence=0.95):
    mean = data.mean()
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, ci

def da_calculate_weighted_mean(values, weights):
    return np.average(values, weights=weights)

def da_get_line_colors(feature, is_pry1):
    WT_DARK = "#666666"
    if not is_pry1:
        ci_color = mean_color = WT_DARK
        ci_width = 1.35
        mean_width = 0.7
        return ci_color, mean_color, ci_width, mean_width
    if feature.startswith("DD"):
        ci_color = mean_color = "#E31A1C"
    elif feature.startswith("DA"):
        ci_color = mean_color = "#33a02c"
    elif feature.startswith("DB"):
        ci_color = mean_color = "#E69F00"
    elif "SAB" in feature:
        ci_color = mean_color = "cyan"
    else:
        ci_color = mean_color = "black"
    ci_width = 3.5 if is_pry1 else 1.35
    mean_width = 4.5 if is_pry1 else 0.7
    return ci_color, mean_color, ci_width, mean_width

def da_prepare_traces(df_melted, fixed_y_value, fixed_y_value_db, fixed_y_value_da8, is_pry1, is_background=False):
    traces = []
    means = {}
    y_values = {}
    ci_levels = {}
    jitter_strength = 0.6
    for feature, group_data in df_melted.groupby("Feature"):
        if feature in ["SABVR", "SABD"]:
            continue
        marker_color = da_determine_color(feature, is_pry1, is_background, for_marker=True)
        y_base = da_determine_y_values(feature, fixed_y_value, fixed_y_value_db, fixed_y_value_da8)
        values = group_data["Value"].dropna()
        if len(values) == 0:
            continue
        value_counts = values.value_counts()
        weights = values.map(value_counts)
        mean_value = da_calculate_weighted_mean(values, weights)
        mean_value, ci = da_calculate_confidence_interval(values)
        if is_pry1:
            marker_trace = go.Scatter(
                y=np.random.normal(y_base, jitter_strength, len(group_data)),
                x=group_data["Value"],
                mode="markers",
                marker=dict(color=marker_color, size=3),
                opacity=0.24,
                name=feature,
                showlegend=False,
            )
            traces.append(marker_trace)
        ci_color, mean_color, ci_width, mean_width = da_get_line_colors(feature, is_pry1)
        ci_y_offset = 1.6 if is_pry1 else -1.6
        ci_level = y_base + ci_y_offset
        ci_levels[feature] = ci_level
        mean_trace = go.Scatter(
            x=[mean_value, mean_value],
            y=[y_base, ci_level],
            mode="lines",
            line=dict(color=mean_color, width=mean_width),
            showlegend=False,
            name=f"{feature} mean",
        )
        ci_trace = go.Scatter(
            x=[mean_value - ci, mean_value + ci],
            y=[ci_level, ci_level],
            mode="lines",
            line=dict(color=ci_color, width=ci_width),
            showlegend=False,
            name=f"{feature} CI",
        )
        traces.append(mean_trace)
        traces.append(ci_trace)
        means[feature] = mean_value
        y_values[feature] = y_base
    return traces, means, y_values, ci_levels

def main_multi(wt_file, mt_files):
    wt_melted, wt_numeric = da_load_and_prepare_data(wt_file, "WT")
    n_genotypes = len(mt_files) + 1
    total_rows = 3 * n_genotypes

    master_fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.01)
    genotype_annotations = []

    fixed_y_value = 0.1
    fixed_y_value_db = 0.1
    fixed_y_value_da8 = 0.1
    wt_traces, wt_means, y_values_wt, ci_levels_wt = da_prepare_traces(
        wt_melted, fixed_y_value, fixed_y_value_db, fixed_y_value_da8,
        is_pry1=True, is_background=False
    )
    for trace in wt_traces:
        if "DD" in trace.name:
            master_fig.add_trace(trace, row=1, col=1)
        elif "DB" in trace.name:
            master_fig.add_trace(trace, row=2, col=1)
        elif "DA" in trace.name:
            master_fig.add_trace(trace, row=3, col=1)
    genotype_annotations.append((2, "WT"))

    for ref_val in [0, 100]:
        for r in range(1, 4):
            master_fig.add_shape(
                type="line",
                x0=ref_val, x1=ref_val,
                y0=0, y1=1,
                xref="x"+str(r),
                yref="paper",
                line=dict(color="black", width=1.5, dash="solid")
            )

    for i, (mt_file, mt_name) in enumerate(mt_files, start=1):
        mt_melted, mt_numeric = da_load_and_prepare_data(mt_file, "MT")
        common_columns = wt_numeric.columns.intersection(mt_numeric.columns)
        df_wt_common = wt_numeric[common_columns]
        df_mt_common = mt_numeric[common_columns]
        sig_results = da_perform_conditional_t_tests(df_wt_common, df_mt_common)
        filtered_results = {node: sig_results[node] for node in sig_results if abs(sig_results[node][1]) >= 0.7}

        wt_traces, wt_means, y_values_wt, ci_levels_wt = da_prepare_traces(
            wt_melted, fixed_y_value, fixed_y_value_db, fixed_y_value_da8,
            is_pry1=False, is_background=False
        )
        mt_traces, mt_means, y_values_mt, ci_levels_mt = da_prepare_traces(
            mt_melted, fixed_y_value, fixed_y_value_db, fixed_y_value_da8,
            is_pry1=True, is_background=False
        )
        arrow_features = filtered_results
        all_sig_features = {node: sig_results[node][0] for node in sig_results}

        row_offset = i * 3
        for trace in wt_traces + mt_traces:
            if "DD" in trace.name:
                master_fig.add_trace(trace, row=row_offset+1, col=1)
            elif "DB" in trace.name:
                master_fig.add_trace(trace, row=row_offset+2, col=1)
            elif "DA" in trace.name:
                master_fig.add_trace(trace, row=row_offset+3, col=1)
        for ref_val in [0, 100]:
            for r in range(1, 4):
                master_fig.add_shape(
                    type="line",
                    x0=ref_val, x1=ref_val,
                    y0=0, y1=1,
                    xref="x"+str(row_offset+r),
                    yref="paper",
                    line=dict(color="black", width=1.5, dash="solid")
                )
        arrow_delta = 0.75
        arrow_asterisk_offset = 3.9
        for feature, (sig, eff, t_stat, df_val) in arrow_features.items():
            if feature in wt_means and feature in mt_means:
                wt_mean = wt_means[feature]
                mt_mean = mt_means[feature]
                ci_level = ci_levels_wt.get(feature, y_values_wt[feature] + (1.6 if feature.startswith("DA") else -1.6))
                if feature.startswith("DD"):
                    r = row_offset+1
                    arrow_color = "#E31A1C"
                elif feature.startswith("DB"):
                    r = row_offset+2
                    arrow_color = "#E69F00"
                elif feature.startswith("DA"):
                    r = row_offset+3
                    arrow_color = "#33a02c"
                else:
                    r = row_offset+1
                    arrow_color = "rgba(0,139,0,0.7)"
                arrow_width = 3 if feature in ["DA8", "DA9"] else 4
                arrow_y = ci_level + arrow_delta
                master_fig.add_annotation(
                    x=mt_mean,
                    y=arrow_y,
                    ax=wt_mean,
                    ay=arrow_y,
                    xref="x"+str(r),
                    yref="y"+str(r),
                    axref="x"+str(r),
                    ayref="y"+str(r),
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=arrow_width,
                    arrowcolor=arrow_color,
                    text=""
                )
                if sig != "***":
                    master_fig.add_annotation(
                        x=mt_mean,
                        y=ci_level + arrow_asterisk_offset,
                        xref="x"+str(r),
                        yref="y"+str(r),
                        text=f"<b>{sig}</b>",
                        showarrow=False,
                        font=dict(color="black", size=21)
                    )
        for feature, sig in all_sig_features.items():
            if feature not in arrow_features and feature in wt_means and feature in mt_means:
                if feature.startswith("DD"):
                    r = row_offset+1
                elif feature.startswith("DB"):
                    r = row_offset+2
                else:
                    r = row_offset+3
                x_coord = mt_means[feature]
                ci_level = ci_levels_wt.get(feature, y_values_wt[feature] + (1.6 if feature.startswith("DA") else -1.6))
                master_fig.add_annotation(
                    x=x_coord,
                    y=ci_level + arrow_asterisk_offset,
                    xref="x"+str(r),
                    yref="y"+str(r),
                    text=f"<b>{sig}</b>",
                    showarrow=False,
                    font=dict(color="black", size=21)
                )
        genotype_annotations.append((row_offset+2, mt_name))
        if i < n_genotypes - 1:
            master_fig.add_shape(
                type="line",
                x0=0, x1=1,
                y0=0, y1=0,
                xref="paper",
                yref="paper",
                line=dict(color="black", width=2, dash="dash")
            )
    all_values = []
    for trace in master_fig.data:
        if isinstance(trace, go.Scatter) and trace.x is not None:
            all_values.extend(trace.x)
    x_min, x_max = min(all_values), max(all_values)
    x_range = [x_min - 0.1*(x_max - x_min), x_max + 0.1*(x_max - x_min)]
    for r in range(1, total_rows+1):
        master_fig.update_xaxes(
            range=x_range, row=r, col=1,
            showgrid=False, zeroline=True,
            zerolinecolor="black", zerolinewidth=2,
            tickmode="array",
            tickvals=[-8, -5, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            ticktext=["-8", "-5", "0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"],
            tickfont=dict(size=16)
        )
        master_fig.update_yaxes(
            showticklabels=False, showgrid=False,
            zeroline=False, row=r, col=1
        )
    for (row_num, name) in genotype_annotations:
        master_fig.add_annotation(
            x=-9,
            y=0.5,
            xref="x"+str(row_num),
            yref="y"+str(row_num),
            text=f"<b><i>{name}</i></b>",
            showarrow=False,
            font=dict(size=19),
            xanchor="right",
            textangle=0
        )
    def get_y_domain(row):
        if row == 1:
            return master_fig.layout.yaxis.domain
        else:
            return master_fig.layout["yaxis"+str(row)].domain

    for i in range(n_genotypes - 1):
        row_bottom = i * 3 + 3
        row_top = (i+1) * 3 + 1
        dom_bottom = get_y_domain(row_bottom)
        dom_top = get_y_domain(row_top)
        y_line = (dom_bottom[1] + dom_top[0]) / 2
        master_fig.add_shape(
            type="line",
            x0=0, x1=1,
            y0=y_line, y1=y_line,
            xref="paper",
            yref="paper",
            line=dict(color="black", width=2, dash="dot")
        )
    master_fig.update_layout(
        plot_bgcolor="rgb(255, 255, 255)",
        paper_bgcolor="rgb(255, 255, 255)",
        showlegend=False,
        height=255 * n_genotypes,
        width=1550,
        title=""
    )
    return master_fig

st.set_page_config(page_title="VNC-Quant/ VNC-Dist", layout="wide")
st.title("2. VNC-Quant/ VNC-Dist")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #87CEFA; }
    [data-testid="stSidebar"] .css-1d391kg { color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Input Data")
csv_files = st.sidebar.file_uploader("Select all CSVs", type=["csv"], accept_multiple_files=True)
mask_files = st.sidebar.file_uploader("Select all PNGs", type=["png"], accept_multiple_files=True)

logos_col1, logos_col2 = st.sidebar.columns(2)
script_path = Path(__file__).parent.resolve()
logo1_path = script_path / "assets" / "TOH.png"
logo2_path = script_path / "assets" / "uOttawaMed.png"

def get_image_base64(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def display_logo(column, logo_path, top_margin=0):
    if logo_path.exists():
        encoded = get_image_base64(logo_path)
        column.markdown(
            f"""
            <div style="margin-top: {top_margin}px;">
                <img src="data:image/png;base64,{encoded}" width="150">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        column.warning(f"Logo not found: {logo_path}")

with logos_col1:
    display_logo(logos_col1, logo1_path)
with logos_col2:
    display_logo(logos_col2, logo2_path, top_margin=35)

st.sidebar.markdown("")

def display_centered_text(text):
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; font-weight: bold; color: black;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

results_container = st.container()

if st.sidebar.button("Process"):
    if not csv_files or not mask_files:
        st.sidebar.error("Upload both CSV and PNG files.")
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            projection_images_dir = os.path.join(temp_dir, "projection_images")
            relative_distances_dir = os.path.join(temp_dir, "relative_distances")
            os.makedirs(projection_images_dir, exist_ok=True)
            os.makedirs(relative_distances_dir, exist_ok=True)
            csv_mapping = {os.path.splitext(f.name)[0]: f for f in csv_files}
            mask_mapping = {os.path.splitext(f.name)[0]: f for f in mask_files}
            common_basenames = set(csv_mapping.keys()).intersection(set(mask_mapping.keys()))
            if not common_basenames:
                st.error("No matching CSV and PNG mask file pairs found based on filenames")
            else:
                total_files = len(common_basenames)
                progress_bar = st.progress(0)
                progress_text = st.empty()
                results = []
                for idx, base_name in enumerate(common_basenames, 1):
                    csv_file = csv_mapping[base_name]
                    mask_file = mask_mapping[base_name]
                    try:
                        worm_measure = WormMeasure(
                            image_file=mask_file,
                            csv_file=csv_file,
                            projection_images_dir=projection_images_dir,
                            relative_distances_dir=relative_distances_dir,
                        )
                        results.append((worm_measure.output_image_path, worm_measure.output_csv_path))
                        progress_text.text(f"Processed {idx}/{total_files}")
                    except Exception as e:
                        st.error(f"Error processing {base_name}: {e}")
                    progress_bar.progress(idx / total_files)
                if results:
                    rd_csv_paths = [pair[1] for pair in results if pair[1].endswith("_RD.csv")]
                    concatenated_pivots = []
                    for rd_csv in rd_csv_paths:
                        try:
                            pivot_data = process_csv(rd_csv)
                            pivot_data.insert(0, "filename", os.path.basename(rd_csv))
                            concatenated_pivots.append(pivot_data)
                        except Exception as e:
                            print(f"Error pivoting {rd_csv}: {e}")
                    if concatenated_pivots:
                        all_data = pd.concat(concatenated_pivots, ignore_index=True)
                        pivot_csv_path = os.path.join(relative_distances_dir, "concatenated_RD.csv")
                        all_data.to_csv(pivot_csv_path, index=False)
                        st.session_state["all_data"] = all_data
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for root, dirs, files in os.walk(projection_images_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.join("projection_images", file)
                                zip_file.write(file_path, arcname)
                        for root, dirs, files in os.walk(relative_distances_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.join("relative_distances", file)
                                zip_file.write(file_path, arcname)
                    zip_buffer.seek(0)
                    st.success("Batch processing completed!")
                    st.download_button(
                        label="Download Results as ZIP",
                        data=zip_buffer,
                        file_name="VNC-Quaant_Results.zip",
                        mime="application/zip",
                    )
                    with results_container:
                        st.subheader("Normalized Relative Distances:")
                        num_results = len(results)
                        for i in range(0, num_results, 2):
                            cols = st.columns(2)
                            for j in range(2):
                                if i + j < num_results:
                                    output_image, output_csv = results[i + j]
                                    image_name = os.path.basename(output_image)
                                    csv_name = os.path.basename(output_csv)
                                    with cols[j]:
                                        try:
                                            with Image.open(output_image) as img:
                                                new_size = (int(img.width * 0.8), int(img.height * 0.8))
                                                resized_img = img.resize(new_size, resample_method)
                                                st.image(resized_img, caption=image_name)
                                        except Exception as e:
                                            st.error(f"Error displaying image {image_name}: {e}")
                                        st.write(f"**RD:** {csv_name}")
                                        try:
                                            df_display = pd.read_csv(output_csv)
                                            columns_to_exclude = ["Cumulative Length DA", "Cumulative Length DD", "Cumulative Length DB"]
                                            df_display = df_display.drop(columns=[col for col in columns_to_exclude if col in df_display.columns], errors="ignore")
                                            st.dataframe(df_display, height=300)
                                        except Exception as e:
                                            st.error(f"Error displaying CSV {csv_name}: {e}")

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

tabs = st.tabs(["Relative distances", "Dot", "Arrow", "Violin-Box", "Ranked position", "QQ-Shapiro"])

with tabs[0]:
    st.header("Relative position")
    st.write("See the processed images and RD CSVs above.")

with tabs[1]:
    st.header("Dot")
    all_data = st.session_state.get("all_data")
    if all_data is None or all_data.empty:
        st.write("No data available for plotting.")
    else:
        df = all_data.copy()
        df["Type"] = "WT"
        df_melted = df.melt(id_vars="Type", var_name="Feature", value_name="Value")
        df_melted["Value"] = pd.to_numeric(df_melted["Value"], errors="coerce")
        df_melted = df_melted[~df_melted["Feature"].str.contains("SAB", case=False, na=False)]
        df_melted = df_melted[~df_melted["Feature"].str.contains("rectum", case=False, na=False)]
        fixed_y_value = 0
        fixed_y_value_db = 0
        fixed_y_value_da8 = 0
        wt_traces, wt_means, y_values_wt = prepare_traces(df_melted, fixed_y_value, fixed_y_value_db, fixed_y_value_da8, is_pry1=True, is_background=False)
        fig = create_subplots(wt_traces, fixed_y_value, fixed_y_value_db, fixed_y_value_da8, plot_bgcolor="rgb(255, 255, 255)")
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("Arrow")
    st.markdown("Upload your Ctrl and at least one mutant CSV file. Enter the corresponding annotation for each dataset.")
    wt_file = st.file_uploader("Upload Ctrl CSV", type=["csv"], key="wt_arrow")
    wt_annotation = st.text_input("Ctrl Annotation", value="WT", key="wt_annot")
    mt_files = st.file_uploader("Upload Mutant CSV(s)", type=["csv"], accept_multiple_files=True, key="mt_arrow")
    mt_annotations = []
    if mt_files:
        st.markdown("### Enter Mutant Annotations")
        for idx, mt_file in enumerate(mt_files):
            annot = st.text_input(f"Annotation for {mt_file.name}:", value=f"MT{idx+1}", key=f"mt_annot_{idx}")
            mt_annotations.append(annot)
    if st.button("Run"):
        if not wt_file:
            st.error("Please upload a Ctrl CSV file.")
        elif not mt_files:
            st.error("Please upload at least one mutant CSV file.")
        elif any([ann.strip() == "" for ann in mt_annotations]):
            st.error("Please provide annotations for all mutant datasets.")
        else:
            mt_list = list(zip(mt_files, mt_annotations))
            try:
                fig_arrow = main_multi(wt_file, mt_list)
                st.plotly_chart(fig_arrow, use_container_width=True)
                st.success("Neuron displacements completed!")
            except Exception as e:
                st.error(f"Error in Arrow analysis: {e}")

with tabs[3]:
    st.header("Violin-Box")
    st.markdown("Upload one or more CSV files and provide the genotype names to create a combined violin & box plot.")
    violin_csv_files = st.file_uploader("Upload CSV(s) for Violin-Box Plot", type=["csv"], accept_multiple_files=True, key="violin_csv")
    if violin_csv_files:
        genotype_names = []
        st.markdown("### Enter genotype names")
        for idx, file in enumerate(violin_csv_files):
            default_label = f"Genotype {idx+1}"
            gt = st.text_input(f"Genotype for {file.name}:", value=default_label, key=f"gt_{idx}")
            genotype_names.append(gt)
        if st.button("Generate Violin Box Plot"):
            fig_violin = violin_box_plot(violin_csv_files, genotype_names)
            if fig_violin:
                st.pyplot(fig_violin)

with tabs[4]:
    st.header("Ranked position")
    st.markdown("Upload one or more CSV files.")
    ranked_csv_files = st.file_uploader("Upload CSV(s) for Ranked Position", type=["csv"], accept_multiple_files=True, key="ranked_csv")
    if ranked_csv_files:
        titles = []
        st.markdown("### Enter genotype name")
        for idx, file in enumerate(ranked_csv_files):
            default_title = f"Ranked Plot {idx+1}"
            title_input = st.text_input(f"Title for {file.name}:", value=default_title, key=f"rank_title_{idx}")
            titles.append(title_input)
        if st.button("Run", key="ranked_run_button"):
            datasets = []
            for file_obj, title in zip(ranked_csv_files, titles):
                try:
                    file_obj.seek(0)
                    df_ranked = process_csv_file(file_obj)
                    datasets.append((df_ranked, title))
                except Exception as e:
                    st.error(f"Error processing {file_obj.name}: {e}")
            if datasets:
                n_plots = len(datasets)
                cols = 3
                rows = math.ceil(n_plots / cols)
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
                if rows == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()
                for idx, (df_ranked, title) in enumerate(datasets):
                    create_bar_plot_on_ax(df_ranked, title, axes[idx])
                for idx in range(n_plots, len(axes)):
                    axes[idx].axis("off")
                plt.tight_layout()
                st.pyplot(fig)

with tabs[5]:
    st.header("QQ-Shapiro")
    st.markdown("Upload one or more CSV files and provide the corresponding genotype names.")
    qq_csv_files = st.file_uploader("Upload CSV(s) for QQ-Shapiro", type=["csv"], accept_multiple_files=True, key="qq_csv")
    if qq_csv_files:
        qq_genotype_names = []
        st.markdown("### Enter Genotype Names")
        for idx, file in enumerate(qq_csv_files):
            default_label = f"Genotype {idx+1}"
            gt = st.text_input(f"Genotype for {file.name}:", value=default_label, key=f"qq_gt_{idx}")
            qq_genotype_names.append(gt)
        if st.button("Generate QQ-Shapiro Plot"):
            fig_qq, shapiro_df = qq_shapiro_analysis(qq_csv_files, qq_genotype_names)
            if fig_qq is not None:
                st.pyplot(fig_qq)
                csv_data = shapiro_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Shapiro-Wilk Results",
                    data=csv_data,
                    file_name="Shapiro_Wilk_Results.csv",
                    mime="text/csv",
                )
            else:
                st.error("Failed to generate QQ-Shapiro plot.")
