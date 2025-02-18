import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from collections import deque
import numpy as np
import rawpy
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import multiprocessing.dummy 
from matplotlib.figure import Figure
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_laplace
import cv2
import subprocess
import sys

class CMOSBack:
    def __init__(self, console_log_callback, update_image_viewer_callback, cancel_event):
        self.console_log = console_log_callback
        self.update_image_viewer = update_image_viewer_callback
        self.cancel_event = cancel_event
        self.histogram_canvas = None
        self.viewer_frame = None

    def save_matrices(self, output_dir, mean_matrix, std_matrix):
        file_path = os.path.join(output_dir, "matrices.npz")
        np.savez(file_path, mean_matrix=mean_matrix, std_matrix=std_matrix)
        self.console_log(f"Matrices saved to {file_path}.")

    def load_matrices(self, output_dir):
        file_path = os.path.join(output_dir, "matrices.npz")
        if os.path.exists(file_path):
            data = np.load(file_path)
            mean_matrix = data["mean_matrix"]
            std_matrix = data["std_matrix"]
            return mean_matrix, std_matrix
        else:
            return None, None

    def threshold(self, file_paths, k, output_dir):
        mean_matrix, std_matrix = self.load_matrices(output_dir)
        if mean_matrix is not None and std_matrix is not None:
            self.console_log("Using precomputed matrices.")
        else:
            sum_matrix = None
            self.console_log('Calculating mean and variance matrices...')
            for idx, file_path in enumerate(file_paths):
                if self.cancel_event.is_set():
                    return
                with rawpy.imread(file_path) as raw:
                    RW = raw.raw_image_visible.copy().astype(np.float64)
                if sum_matrix is None:
                    n = len(file_paths)
                    K = RW
                    cx = np.zeros_like(RW, dtype=np.float64)
                    cx2 = np.zeros_like(RW, dtype=np.float64)
                    sum_matrix = np.zeros_like(RW, dtype=np.float64)
                cx += RW - K
                cx2 += (RW - K) ** 2
                sum_matrix += RW
                if not self.cancel_event.is_set():
                    self.console_log(f"Processing image {idx + 1}/{n}.")
            mean_matrix = sum_matrix / n
            variance_matrix = (cx2 - (cx**2 / n)) / (n - 1)
            std_matrix = np.sqrt(variance_matrix)
            if not self.cancel_event.is_set():
                self.save_matrices(output_dir, mean_matrix, std_matrix)
        matrix_threshold = mean_matrix + k * std_matrix
        return matrix_threshold, mean_matrix
    
    
    def set_viewer_frame(self, viewer_frame):
        self.viewer_frame = viewer_frame

    def clustering(self, indices, epsilon):
        if len(indices) == 0:
            return [], []

        tree = BallTree(indices, metric='chebyshev')
        labels = np.full(len(indices), -1, dtype=int)
        cluster_id = 0
        visited = set()
        N_el = []

        for i, point in enumerate(indices):
            if i in visited:
                continue

            neighbors = tree.query_radius(point.reshape(1, -1), r=epsilon)[0]
            if len(neighbors) == 1:
                labels[i] = -1
                N_el.append(1)
                continue

            cluster_id += 1
            cluster_points = set(neighbors)
            labels[list(cluster_points)] = cluster_id
            visited.update(cluster_points)
            clustering = deque(cluster_points)
            while clustering:
                current = clustering.popleft()
                new_neighbors = tree.query_radius(indices[current].reshape(1, -1), r=epsilon)[0]
                for neighbor in new_neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        labels[neighbor] = cluster_id
                        clustering.append(neighbor)
            cluster_size = np.sum(labels == cluster_id)

            N_el.append(cluster_size)

        clusters = [indices[labels == cid] for cid in range(1, cluster_id + 1)]
        return clusters, N_el

    def clusteringDBSCAN(self, indices, epsilon, min_pts):
            if len(indices) == 0:
                return [], []

            tree = BallTree(indices, metric='chebyshev')
            labels = np.full(len(indices), -1, dtype=int)
            cluster_id = 0
            visited = set()
            N_el = []

            for i, point in enumerate(indices):
                if i in visited:
                    continue

                neighbors = tree.query_radius(point.reshape(1, -1), r=epsilon)[0]
                if len(neighbors) < min_pts:
                    labels[i] = -1
                    continue

                cluster_id += 1
                cluster_points = set(neighbors)
                labels[list(cluster_points)] = cluster_id
                visited.update(cluster_points)
                clustering = deque(cluster_points)
                while clustering:
                    current = clustering.popleft()
                    new_neighbors = tree.query_radius(indices[current].reshape(1, -1), r=epsilon)[0]
                    if len(new_neighbors) >= min_pts:
                        for neighbor in new_neighbors:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                labels[neighbor] = cluster_id
                                clustering.append(neighbor)
                cluster_size = np.sum(labels == cluster_id)
                N_el.append(cluster_size)

            clusters = [indices[labels == cid] for cid in range(1, cluster_id + 1)]
            return clusters, N_el

    def crop_n_save(self, file_prefix, image, Cluster_pos, output_dir, cmap_chosen, crop_size_chosen):
        half_crop = crop_size_chosen // 2
        for i, cluster in enumerate(Cluster_pos):
            if self.cancel_event.is_set():
                return
            size = len(cluster)
            center = np.mean(cluster, axis=0).astype(int)
            x, y = center
            top, bottom = max(0, x - half_crop), min(image.shape[0], x + half_crop)
            left, right = max(0, y - half_crop), min(image.shape[1], y + half_crop)
            cropped_image = image[top:bottom, left:right]
            crop_path = os.path.join(output_dir, f"{file_prefix}_cluster_{i + 1}_coord({x},{y})_size({size}).tiff")
            plt.imsave(crop_path, cropped_image, cmap=cmap_chosen)
            self.console_log(f"Crop saved at {crop_path}. Coordinates: ({x}, {y}), Size: {size}")
            self.update_image_viewer(cropped_image)

    def crop_n_save_LoG(self, file_prefix, image, event_positions, output_dir, cmap_chosen, crop_size_chosen):
        half_crop = crop_size_chosen // 2
        for i, (x, y) in enumerate(event_positions):
            if self.cancel_event.is_set():
                return
            top, bottom = max(0, x - half_crop), min(image.shape[0], x + half_crop)
            left, right = max(0, y - half_crop), min(image.shape[1], y + half_crop)
            cropped_image = image[top:bottom, left:right]
            crop_path = os.path.join(output_dir, f"{file_prefix}_event_{i + 1}_coord({x},{y}).tiff")
            plt.imsave(crop_path, cropped_image, cmap=cmap_chosen)
            self.console_log(f"Crop saved at {crop_path}. Coordinates: ({x}, {y})")
            self.update_image_viewer(cropped_image)

    def plot_histogram(self, cluster_sizes, output_dir):
        unique_sizes, counts = np.unique(cluster_sizes, return_counts=True)
        total_counts = np.sum(counts)
        fig = Figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        ax.bar(unique_sizes, counts, align='center', width=0.8)
        
        ax.set_xlabel('Size (pixels)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of detections (Number of pixels)')
        ax.set_xticks(np.setdiff1d(unique_sizes, [0]))  

        ax.text(0.5, 0.95, f'Number of detections: {total_counts}', 
                transform=ax.transAxes, ha='center', va='top', fontsize=8, color='black',
               bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        #fig.tight_layout()
        fig.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.15)
        self.histogram_canvas = FigureCanvasTkAgg(fig, master=self.viewer_frame)
        canvas_widget = self.histogram_canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, columnspan=5, sticky="nsew", padx=5, pady=5)
        self.histogram_canvas.draw()

        histogram_path = os.path.join(output_dir, "histogram.png") 
        fig.savefig(histogram_path, format='png', bbox_inches='tight', dpi=300) 
        self.console_log("Histogram saved in the output directory.")
        

        file_name = "detection_data.txt"
        with open(f"{output_dir}/{file_name}", "w") as file:
            file.write("Size\tCounts\n")  
            for size, count in zip(unique_sizes, counts):
                file.write(f"{size}\t{count}\n")
            file.write(f"\nTotal detections: {total_counts}\n")  
            file.flush()

        self.console_log(f"Results saved in {file_name}.")

    def clear_histogram(self):
        if hasattr(self, "histogram_canvas") and self.histogram_canvas is not None:
            self.histogram_canvas.get_tk_widget().destroy()
            self.histogram_canvas = None


class CMOSdec:
    def __init__(self, root):
        self.root = root
        self.cancel_event = multiprocessing.dummy.Event()
        self.root.title("TRACED for CMOS: Transient Radiation Anomaly and Cosmic-ray Event Detector")
        self.root.iconbitmap("images/iconS.ico")
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.epsilon = tk.IntVar(value=10)
        self.min_pts = tk.IntVar(value=3)
        self.k_value = tk.IntVar(value=6)
        self.method = tk.StringVar(value="Method 1")
        self.sigma = tk.IntVar(value=6)
        self.offset = tk.IntVar()
        self.colormaps = sorted(plt.colormaps())
        self.colormap_selected = tk.StringVar(value="gray")
        self.subtract_mean_var = tk.BooleanVar(value=True)
        self.crop_size = tk.IntVar(value=50)
        self.processing = False
        self.back = CMOSBack(self.console_log, self.update_image_viewer, self.cancel_event)
        self.stack_var = tk.BooleanVar(value=False)
        self.stacking_method_var = tk.StringVar()
        self.create_widgets()
        
        

    def create_widgets(self):
        #Directory Frame
        dir_frame = ttk.LabelFrame(root, text="Directories", padding=10)
        dir_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
    
        ttk.Label(dir_frame, text="Input Directory:").grid(row=0, column=0, sticky="w")
        ttk.Entry(dir_frame, textvariable=self.input_dir, width=70).grid(row=0, column=1, padx=5)
        ttk.Button(dir_frame, text="Browse", command=self.browse_input).grid(row=0, column=2, padx=5,pady=10)
    
        ttk.Label(dir_frame, text="Output Directory:").grid(row=1, column=0, sticky="w")
        ttk.Entry(dir_frame,textvariable=self.output_dir, width=70).grid(row=1, column=1, padx=5)
        ttk.Button(dir_frame, text="Browse", command=self.browse_output).grid(row=1, column=2, padx=5)


        # Threshold Frame
        thres_frame = ttk.LabelFrame(root, text="Thresholding", padding=10)
        thres_frame.grid(row=1, column=0, columnspan=1, rowspan=1, padx=10, pady=10, sticky="ew")

        ttk.Label(thres_frame, text="Epsilon value (pixels):").grid(row=0, column=3, sticky="w", padx=5)
        self.epsentry = ttk.Entry(thres_frame, textvariable=self.epsilon, width=10)
        self.epsentry.grid(row=0, column=4, sticky="w", pady=5)

        epsentry_ttp = CreateToolTip(self.epsentry, 'Value for the eps-neighborhood search (In Chebyshev metric)')
        
        ttk.Label(thres_frame, text="K Value (sigmas):").grid(row=0, column=0, sticky="w", padx=5)
        self.k_valueentry = ttk.Entry(thres_frame, textvariable=self.k_value, width=10)
        self.k_valueentry.grid(row=0, column=1, sticky="w", pady=5)

        k_valueentry_ttp = CreateToolTip(self.k_valueentry, 'Insert the threshold of detection for events, being k such that I(x,y)>Im(x,y)+k*sigma')
        

        ttk.Label(thres_frame, text="Sigma (LoG):").grid(row=1, column=0, sticky="w", padx=5)
        self.sigmaentry = ttk.Entry(thres_frame, textvariable=self.sigma,state="disabled", width=10)
        self.sigmaentry.grid(row=1, column=1, sticky="w", pady=5)
        sigmaentry_ttp = CreateToolTip(self.sigmaentry, 'Sigma value of the LoG kernel')

        
        ttk.Label(thres_frame, text="Min. points (DBSCAN):").grid(row=1, column=3, sticky="w", padx=5)
        self.min_pts_entry = ttk.Entry(thres_frame, textvariable=self.min_pts,state="disabled", width=10)
        self.min_pts_entry.grid(row=1, column=4, sticky="w", pady=5)
        min_pts_entry_ttp = CreateToolTip(self.min_pts_entry, 'Minimum number of points in a epsilon neighborhood for a DBSCAN core point')


        #Method frame
        method_frame = ttk.LabelFrame(self.root, text="Method", padding=10) 

        method_frame.grid(row=2, column=0, columnspan=1, rowspan=1, padx=10, pady=10, sticky="w")  
        ttk.Label(method_frame, text="Method:").pack(side="left") 
        self.seqradio = ttk.Radiobutton(method_frame, text="Sequential", variable=self.method, value="Method 1")
        self.seqradio.pack(side="left", padx=5)
        self.DBSCANradio = ttk.Radiobutton(method_frame, text="DBSCAN", variable=self.method, value="Method 2")
        self.DBSCANradio.pack(side="left", padx=5)
        self.LoGradio = ttk.Radiobutton(method_frame, text="LoG", variable=self.method, value="Method 3")
        self.LoGradio.pack(side="left", padx=5)

        seqradio_ttp = CreateToolTip(self.seqradio, 'Select a point and consider any point in an eps-neighborhood of every detected point as an associated event.')
        DBSCANradio_ttp = CreateToolTip(self.DBSCANradio, 'Detect and categorize events as a cluster on the basis of the DBSCAN method.')
        LoGradio_ttp = CreateToolTip(self.LoGradio, 'LoG detection (Experimental): Uses the convolution of the image with a Laplacian of Gaussian kernel for detection.')

        self.method.trace_add("write", lambda *args: self.on_method_change())

        #Stacking frame
        stacking_frame = ttk.LabelFrame(self.root, text="Stacking", padding=10) 

        stacking_frame.grid(row=2, column=1, columnspan=1, rowspan=1, padx=10, pady=10, sticky="w")
        ttk.Label(stacking_frame, text="Method:").pack(side="left") 

        self.method_stacking = ttk.Combobox(stacking_frame, values=["Mean Stacking", "Sum Stacking"],textvariable=self.stacking_method_var, width=13, height=3)
        self.method_stacking.pack(side="left", padx=5) 
        stacking_ttp = CreateToolTip(self.method_stacking, 'Select the type of stacking to be used. Sum stacking will just integrate all frames, mean stacking will normalize by the number of frames stacked.')

        self.stacking_checkbox = ttk.Checkbutton(stacking_frame, text="Stack", variable=self.stack_var, command=self.stack_method)
        self.stacking_checkbox.pack() 
        stacking_ttp = CreateToolTip(self.stacking_checkbox, '(Experimental) Stack the frames, creating a resulting image presenting the overlay of all detections.')

        # Visualization Frame
        Visualizer = ttk.LabelFrame(self.root, text="Visualization options", padding=17)  
        Visualizer.grid(row=1, column=1, rowspan=1,columnspan=1, padx=10 )   
        self.crop_size.trace_add("write", self.update_crop_size)

        ttk.Label(Visualizer, text="Crop sqr. side (pixels):").grid(row=0, column=0, padx=5)
        self.crop_size_entry = ttk.Entry(Visualizer, textvariable=self.crop_size, width=10)
        self.crop_size_entry.grid(row=0, column=1, sticky="w", padx=5)
        self.crop_size_entry.bind("<FocusOut>", self.update_crop_size)

        self.subtract_mean_checkbox = ttk.Checkbutton(Visualizer, text="Mean subtraction", 
            variable=self.subtract_mean_var)
        self.subtract_mean_checkbox.grid(row=1, column=0, sticky="w",pady=5)
        meansub_ttp = CreateToolTip(self.subtract_mean_checkbox, 'Subtracts from each crop image the calculated mean, working as a filter')

        #Process Frame
        Process_frame = ttk.Frame(self.root)
        Process_frame.grid(row=3, column=0, rowspan=1, columnspan=2, padx=5, pady=5, sticky="w")
        self.start_button = ttk.Button(Process_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(side="left", padx=50)

        self.cancel_button = ttk.Button(Process_frame, text="Cancel Processing", command=self.cancel_processing, state="disabled")
        self.cancel_button.pack(side="left", padx=50)

        ttk.Button(Process_frame, text="Open Output Folder", command=self.open_output_folder).pack(side="left", padx=50)

        # Console frame
        console_frame = ttk.LabelFrame(self.root, text="Output:", padding=10)
        console_frame.grid(row=4, column=0, columnspan=4, sticky="nsew", padx=10, pady=10)
        self.console_text = tk.Text(console_frame, wrap="word", height=7, state="normal", bg="dark gray",fg="black")
        self.console_text.pack(fill="both", expand=True, padx=10, pady=5)
        self.console_text.bind("<Key>", lambda e: "break")  
        self.console_text.bind("<Button-1>", lambda e: "break")  

        # Image Viewer frame
        fig, self.ax = plt.subplots(figsize=(4.7, 3.8))
        self.ax.grid(False)
        Viewer_frame = ttk.LabelFrame(self.root, text="Display", padding=0)
        Viewer_frame.grid(row=0, column=2, columnspan=2, rowspan=4, sticky="nsew", padx=10, pady=10)
        self.canvas = FigureCanvasTkAgg(fig, master=Viewer_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, columnspan=5, sticky="nsew", padx=5, pady=5)
        self.cmap_selector = ttk.Combobox(Viewer_frame, values=self.colormaps,textvariable=self.colormap_selected, width=3, height=3)
        self.cmap_selector.grid(row=1, column=0, columnspan=1, sticky="nsew", padx=5, pady=5)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.back.set_viewer_frame(Viewer_frame)


    def stack_method(self):
        if self.stack_var.get() == True:
            self.method.set("stack")
            self.seqradio.config(state="disabled")
            self.DBSCANradio.config(state="disabled")
            self.LoGradio.config(state="disabled")
            self.min_pts_entry.config(state="disabled")
            self.subtract_mean_checkbox.config(state="disabled")
            self.k_valueentry.config(state="disabled")
            self.epsentry.config(state="disabled")
            self.sigmaentry.config(state="disabled")
            self.crop_size_entry.config(state="disabled")
        else:
            self.method.set("Method 1")
            self.seqradio.config(state="normal")
            self.DBSCANradio.config(state="normal")
            self.LoGradio.config(state="normal")
            self.crop_size_entry.config(state="normal")
            

    def on_method_change(self):
        if self.method.get() == "Method 1":
            self.min_pts_entry.config(state="disabled")
            self.subtract_mean_checkbox.config(state="normal")
            self.k_valueentry.config(state="normal")
            self.epsentry.config(state="normal")
            self.sigmaentry.config(state="disabled")
        elif self.method.get() == "Method 2":
            self.min_pts_entry.config(state="normal")
            self.subtract_mean_checkbox.config(state="normal")
            self.k_valueentry.config(state="normal")
            self.epsentry.config(state="normal")
            self.sigmaentry.config(state="disabled")
        elif self.method.get() == "Method 3":
            self.min_pts_entry.config(state="disabled")
            self.subtract_mean_checkbox.config(state="disabled")
            self.k_valueentry.config(state="disabled")
            self.epsentry.config(state="disabled")
            self.sigmaentry.config(state="normal")
        
            
    def update_offset_from_black_level(self, input_dir):
        file_paths = glob.glob(os.path.join(input_dir.get(), '*'))
        if not file_paths:
            self.console_log("No RAW files found in the input directory.")
            return

        first_file = file_paths[0]
        try:
            with rawpy.imread(first_file) as raw:
                black_level = int(np.mean(raw.black_level_per_channel))
            self.offset.set(black_level)  
            self.console_log(f"Black level of {os.path.basename(first_file)} : {black_level} ADU.")
        except Exception as e:
            self.console_log(f"Error reading black level: {e}")
    
    def process_images(self):
        try:
            input_dir = self.input_dir.get()
            output_dir = self.output_dir.get()
            epsilon = self.epsilon.get()
            k = self.k_value.get()
            min_pts = self.min_pts.get()
            cmap_chosen = self.colormap_selected.get()
            crop_size_chosen = self.crop_size.get()
            method = self.method.get()
            file_paths = glob.glob(os.path.join(input_dir, '*'))
            if not file_paths:
                self.console_log("No RAW files found in the input directory.")
                return

            self.cancel_event.clear()
            matrix_threshold, mean_matrix = self.back.threshold(file_paths, k, output_dir)
            cluster_sizes_all = []
            

            for idx, file_path in enumerate(file_paths):
                if self.cancel_event.is_set():
                    return  
                with rawpy.imread(file_path) as raw:
                    RAW_cp = raw.raw_image_visible.copy().astype(np.float64)

                indices = np.argwhere(RAW_cp > matrix_threshold)
                if method == "Method 1":
                    Cluster_pos, cluster_sizes = self.back.clustering(indices, epsilon)
                else:
                    Cluster_pos, cluster_sizes = self.back.clusteringDBSCAN(indices, epsilon, min_pts)

                cluster_sizes_all.extend(cluster_sizes)
                file_prefix = os.path.basename(file_path).split('.')[0]
                if self.subtract_mean_var.get():
                    diff_cap = np.maximum(RAW_cp - mean_matrix, 0)
                else:
                    diff_cap = RAW_cp  
                
                self.back.crop_n_save(file_prefix, diff_cap, Cluster_pos, output_dir, cmap_chosen,crop_size_chosen)
            self.back.clear_histogram() 
            self.back.plot_histogram(cluster_sizes_all, output_dir)
            del RAW_cp
            
            self.console_log("Processing completed.")

        except Exception as e:
            error_message = str(e)
            self.console_log(f"Error: {error_message}")
        finally:
            self.processing = False
            self.start_button.config(state="normal")
            self.cancel_button.config(state="disabled")

    def process_image_LoG(self):
        try:
            input_dir = self.input_dir.get()
            output_dir = self.output_dir.get()
            sigma = self.sigma.get()
            cmap_chosen = self.colormap_selected.get()
            crop_size_chosen = self.crop_size.get()
            file_paths = glob.glob(os.path.join(input_dir, '*'))
            if not file_paths:
                self.console_log("No RAW files found in the input directory.")
                return
            self.cancel_event.clear()
            detections_all = 0
            for idx, file_path in enumerate(file_paths):
                if self.cancel_event.is_set():
                    return  
            
                with rawpy.imread(file_path) as raw:
                    RAW_cp = raw.raw_image_visible.copy().astype(np.float64)
                    
                # Apply gaussian kernel
                laplacian_image = gaussian_laplace(RAW_cp, sigma)
                
                detec_mask = laplacian_image < 0 
            
                # Label connected components
                labeled_image = label(detec_mask, connectivity=2)
            
                # Extract centroids
                event_positions = [tuple(map(int, prop.centroid)) for prop in regionprops(labeled_image)]
                
                file_prefix = os.path.basename(file_path).split('.')[0]
                self.back.crop_n_save_LoG(file_prefix, RAW_cp, event_positions, output_dir, cmap_chosen, crop_size_chosen)
                counts = len(event_positions)
                detections_all += counts
                self.console_log(f"Detected {counts} events using LoG.")
            if not self.cancel_event.is_set():
                self.console_log(f"{detections_all} events detected in total.")
    
        except Exception as e:
            error_message = str(e)
            self.console_log(f"Error: {error_message}")
        finally:
            self.processing = False
            self.start_button.config(state="normal")
            self.cancel_button.config(state="disabled")


    def stack_images(self):
        try:
            input_dir = self.input_dir.get()
            output_dir = self.output_dir.get()
            file_paths = glob.glob(os.path.join(input_dir, '*'))
            method = self.stacking_method_var.get()
            mean_matrix, std_matrix = self.back.load_matrices(output_dir)
            if mean_matrix is not None and std_matrix is not None:
                self.console_log("Using precomputed matrices.")
                offset = mean_matrix
            else:
                self.console_log("No calculated matrix found in the output directory. Calculate the matrices before stacking.")
                return 
            sum_matrix = None
            self.cancel_event.clear()
            for idx, file_path in enumerate(file_paths):
                if self.cancel_event.is_set():
                    return
                with rawpy.imread(file_path) as raw:
                    RW = raw.raw_image_visible.copy().astype(np.float32)
                if sum_matrix is None:
                    n = len(file_paths)
                    sum_matrix = np.zeros_like(RW, dtype=np.float32)
                sum_matrix += np.maximum((RW - offset),0)
                
                #self.update_image_viewer(np.maximum(sum_matrix,0))
                if not self.cancel_event.is_set():
                    self.console_log(f"Stacking image {idx + 1}/{n}.")
            if method == "Mean Stacking":
                stacked_image = sum_matrix/n
            else:
                stacked_image = sum_matrix

            max_value = np.max(stacked_image)
            self.console_log(f"stacked min/max: {int(stacked_image.min())}, {int(stacked_image.max())}")
            if max_value <= 255:
                dtype = np.uint8 
            else:
                dtype = np.uint16  
            output_path = os.path.join(output_dir, "stacked_image.png")
            stacked_image = cv2.normalize(stacked_image, stacked_image, alpha=0, beta=np.iinfo(dtype).max, norm_type=cv2.NORM_MINMAX)
            stacked_image_f = stacked_image.astype(dtype)
            cv2.imwrite(output_path, stacked_image_f)
            self.update_image_viewer(stacked_image_f)
            self.console_log(f"Stacked image saved in {output_path}.")
        except Exception as e:
            error_message = str(e)
            self.console_log(f"Error: {error_message}")
        finally:
            self.processing = False
            self.start_button.config(state="normal")
            self.cancel_button.config(state="disabled")
        

    def on_close(self):
        if self.processing:
            self.cancel_processing()

        self.canvas.get_tk_widget().destroy()
        plt.close('all')
        self.root.destroy()

    def browse_input(self):
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir.set(directory)
            self.update_offset_from_black_level(self.input_dir)

    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)

    def open_output_folder(self):
        if sys.platform == "win32" and not os.path.exists(self.output_dir.get()):
            messagebox.showerror("Error", "Output directory does not exist.")
            return
        if sys.platform == "win32":
            os.startfile(self.output_dir.get())
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", self.output_dir.get()])
        else:  # Linux
            subprocess.run(["xdg-open", self.output_dir.get()])
        

    def limit_console_lines(self):
        MAX_LINES = 20
        num_lines = int(self.console_text.index("end-1c").split(".")[0]) 
        if num_lines > MAX_LINES:
            self.console_text.delete("1.0", f"2.0")  

    def console_log(self, message):
        self.console_text.insert(tk.END, f"{message}\n")
        self.console_text.see(tk.END)
        self.limit_console_lines()
        self.root.update()

    def update_crop_size(self, *args):
        crop_size_value = self.crop_size.get()
        if isinstance(crop_size_value, (int, float)) and crop_size_value > 0:
            crop_size = int(crop_size_value)
            self.console_log(f"Crop size updated to: {crop_size}x{crop_size} pixels")
        else:
            self.console_log("Invalid crop size entered. Please enter a valid positive integer.")


    def update_image_viewer(self, image):
        if hasattr(self, "histogram_canvas"):  
            self.histogram_canvas.get_tk_widget().destroy()
            del self.histogram_canvas
        self.back.clear_histogram()
        selected_cmap = self.colormap_selected.get()
        #self.ax.clear()
        self.ax.imshow(image, cmap=selected_cmap)
        self.canvas.draw()
        

    def start_processing(self):
        if self.processing:
            return 
        
        if not self.input_dir.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Please select both input and output directories.")
            return
        method = self.method.get()
        self.processing = True
        self.start_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.console_log("Processing started...")
        self.processing_thread_pool = multiprocessing.dummy.Pool(9) #TODO fix the threading !!!
        if method == "Method 3":
            self.processing_thread_pool.apply_async(self.process_image_LoG) 
        elif method == "stack":
            self.processing_thread_pool.apply_async(self.stack_images)
        else:
            self.processing_thread_pool.apply_async(self.process_images)
         
        

    def cancel_processing(self):
        if self.processing_thread_pool:
            self.cancel_event.set()
            self.processing_thread_pool.close()
            self.processing_thread_pool.terminate()
            #self.processing_thread_pool.join() #FIXME
            self.processing = False
            self.console_log("Processing canceled.")
            self.start_button.config(state="normal")
            self.cancel_button.config(state="disabled")



class CreateToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.waittime = 500     #miliseconds
        self.wraplength = 180   #pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = ttk.Label(self.tw, text=self.text, justify='left',
                background="#333333",  
                 foreground="white",  relief='solid', borderwidth=1,
                       wraplength = self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.tk.call('source', 'Theme/forest-dark.tcl')
    ttk.Style().theme_use('forest-dark')
    app = CMOSdec(root)
    root.mainloop()
