import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import rawpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from matplotlib.figure import Figure
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_laplace
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class CMOSBack:
    def __init__(self, console_log_callback, update_image_viewer_callback, cancel_event):
        self.console_log = console_log_callback
        self.update_image_viewer = update_image_viewer_callback
        self.cancel_event = cancel_event
        self.histogram_canvas = None
        self.viewer_frame = None

    def save_matrices(self, output_dir, mean_matrix, std_matrix):
        file_path = os.path.join(output_dir, "prcs_data.npz")
        np.savez(file_path, mean_matrix=mean_matrix, std_matrix=std_matrix)
        self.console_log(f"Precomputed data saved in {file_path}.")

    def load_matrices(self, output_dir):
        file_path = os.path.join(output_dir, "prcs_data.npz")
        if os.path.exists(file_path):
            try:
                data = np.load(file_path)
                return data["mean_matrix"], data["std_matrix"]
            except Exception as e:
                self.console_log(f"Failed to load the data: {e}")
                return None, None
        else:
            return None, None

    def threshold(self, file_paths, k, output_dir):
        mean_matrix_cpu, std_matrix_cpu = self.load_matrices(output_dir)

        if mean_matrix_cpu is not None and std_matrix_cpu is not None:
            self.console_log("Using precomputed data.")
            mean_matrix_dev = mean_matrix_cpu
            std_matrix_dev = std_matrix_cpu
        else:
            self.console_log(f"Calculating statistics for event detection...")
            n = len(file_paths)
            if n == 0: return None, None
            
            with rawpy.imread(file_paths[0]) as raw:
                first_image_cpu = raw.raw_image_visible.copy().astype(np.float32)
                K = first_image_cpu

            cx = np.zeros_like(K, dtype=np.float32)
            cx2 = np.zeros_like(K, dtype=np.float32)
            sum_matrix = np.zeros_like(K, dtype=np.float32)

            for idx, file_path in enumerate(file_paths):
                if self.cancel_event.is_set(): return None, None
                with rawpy.imread(file_path) as raw:
                    RW_dev = raw.raw_image_visible.copy().astype(np.float32)
                
                cx += RW_dev - K
                cx2 += (RW_dev - K) ** 2
                sum_matrix += RW_dev
                self.console_log(f"Processing image {idx + 1}/{n}.")

            mean_matrix_dev = sum_matrix / n
            variance_matrix = (cx2 - (cx**2 / n)) / (n - 1)
            std_matrix_dev = np.sqrt(variance_matrix)

            if not self.cancel_event.is_set():
                self.save_matrices(output_dir, mean_matrix_dev, std_matrix_dev)

        matrix_threshold_dev = mean_matrix_dev + k * std_matrix_dev
        final_threshold = matrix_threshold_dev
        final_mean = mean_matrix_dev
        return final_threshold, final_mean

    def set_viewer_frame(self, viewer_frame):
        self.viewer_frame = viewer_frame


    def clusteringDBSCAN(self, indices, epsilon, min_pts):
        if len(indices) == 0:
            return [], []

        db = DBSCAN(eps=epsilon, min_samples=min_pts, metric="chebyshev")
        labels = db.fit_predict(indices)

        clusters = []
        sizes = []

        for label in set(labels):
            if label == -1:
                continue
            cluster = indices[labels == label]
            clusters.append(cluster)
            sizes.append(len(cluster))

        return clusters, sizes

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
            crop_path = os.path.join(output_dir, f"{file_prefix}_cluster_{i + 1}_coord({x},{y})_size({size}).png")
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
            crop_path = os.path.join(output_dir, f"{file_prefix}_event_{i + 1}_coord({x},{y}).png")
            plt.imsave(crop_path, cropped_image, cmap=cmap_chosen)
            self.console_log(f"Crop saved at {crop_path}. Coordinates: ({x}, {y})")
            self.update_image_viewer(cropped_image)

    def plot_histogram(self, cluster_sizes, output_dir):
        unique_sizes, counts = np.unique(cluster_sizes, return_counts=True)
        total_counts = np.sum(counts)
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.bar(unique_sizes, counts, align='center', width=0.8)
        ax.set_xlabel('Size (pixels)'); 
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Detections')
        fig.tight_layout(pad=2)

        ax.set_xticks(np.setdiff1d(unique_sizes, [0]))  

        ax.text(0.5, 0.95, f'Number of detections: {total_counts}', 
                transform=ax.transAxes, ha='center', va='top', fontsize=8, color='black',
               bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        
        self.histogram_canvas = FigureCanvasTkAgg(fig, master=self.viewer_frame)
        self.histogram_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.histogram_canvas.draw()
        
        fig.savefig(os.path.join(output_dir, "histogram.png"), dpi=300)

        
        file_name = "detection_data.txt"
        with open(f"{output_dir}/{file_name}", "w") as file:
            file.write("Size\tCounts\n")  
            for size, count in zip(unique_sizes, counts):
                file.write(f"{size}\t{count}\n")
            file.write(f"\nTotal detections: {total_counts}\n")  
            file.flush()

        self.console_log(f"Results saved in {file_name}.")

    def clear_histogram(self):
        if hasattr(self, "histogram_canvas") and self.histogram_canvas:
            self.histogram_canvas.get_tk_widget().destroy()
            self.histogram_canvas = None

class CMOSdec:
    def __init__(self, root):
        self.root = root
        self.root.title("TRACED: Transient Radiation Anomaly and Cosmic-ray Event Detector")
        base = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base, "images", "iconS.ico")

        self.root.iconbitmap(icon_path)
        self.root.minsize(900, 600)

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.epsilon = tk.IntVar(value=10)
        self.min_pts = tk.IntVar(value=3)
        self.k_value = tk.IntVar(value=6)
        self.method = tk.StringVar(value="DBSCAN")
        self.sigma = tk.IntVar(value=6)
        self.subtract_mean_var = tk.BooleanVar(value=True)
        self.cancel_event = threading.Event()
        self.crop_size = tk.IntVar(value=50)
        self.stack_var = tk.BooleanVar(value=False)
        self.stacking_method_var = tk.StringVar(value="Sum Stacking")
        self.colormap_selected = tk.StringVar(value="gray")
        self.processing = False
        self.back = CMOSBack(self.console_log, self.update_image_viewer, self.cancel_event)

        self.create_widgets()
        


    def create_widgets(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=4) 
        self.root.rowconfigure(1, weight=1) 

        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        main_frame.columnconfigure(0, weight=0) 
        main_frame.columnconfigure(1, weight=1) 
        main_frame.rowconfigure(0, weight=1)

        controls_panel = ttk.Frame(main_frame)
        controls_panel.grid(row=0, column=0, sticky="ns", padx=5)

        self._create_directory_widgets(controls_panel)
        self._create_thresholding_widgets(controls_panel)
        self._create_options_widgets(controls_panel)
        self._create_method_widgets(controls_panel)
        self._create_stacking_widgets(controls_panel)
        self._create_process_widgets(controls_panel)

        self._create_display_widgets(main_frame)
        self._create_console_widgets(self.root)
        self.on_method_change()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_frame(self, parent, text):
        frame = ttk.LabelFrame(parent, text=text, padding=10)
        frame.pack(side="top", fill="x", padx=5, pady=5)
        return frame

    def _create_directory_widgets(self, parent):
        frame = self._create_frame(parent, "Directories")
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="Input Directory:").grid(row=0, column=0, sticky="w", pady=10)
        ttk.Entry(frame, textvariable=self.input_dir).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(frame, text="Browse", command=self.browse_input, width=8).grid(row=0, column=2)
        ttk.Label(frame, text="Output Directory:").grid(row=1, column=0, sticky="w", pady=10)
        ttk.Entry(frame, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(frame, text="Browse", command=self.browse_output, width=8).grid(row=1, column=2)

    def _create_thresholding_widgets(self, parent):
        frame = self._create_frame(parent, "Thresholding")
        frame.columnconfigure((1, 3), weight=1)
        
        ttk.Label(frame, text="K Value:").grid(row=0, column=0, sticky="w")
        self.k_value_entry = ttk.Entry(frame, textvariable=self.k_value, width=8)
        self.k_value_entry.grid(row=0, column=1, padx=5, sticky="ew")
        CreateToolTip(self.k_value_entry, "Threshold for event detection, such that: Signal > Mean + k * Sigma.")

        ttk.Label(frame, text="Epsilon (px):").grid(row=0, column=2, sticky="w", padx=(10,0))
        self.eps_entry = ttk.Entry(frame, textvariable=self.epsilon, width=8)
        self.eps_entry.grid(row=0, column=3, padx=5, sticky="ew")
        CreateToolTip(self.eps_entry, "Radius for neighborhood search in pixels (Chebyshev distance).")

        ttk.Label(frame, text="Sigma (LoG):").grid(row=1, column=0, sticky="w", pady=(5,0))
        self.sigmaentry = ttk.Entry(frame, textvariable=self.sigma, state="disabled", width=8)
        self.sigmaentry.grid(row=1, column=1, padx=5, sticky="ew", pady=(5,0))
        CreateToolTip(self.sigmaentry, "Value of sigma for the Laplacian of Gaussian (LoG) kernel.")

        ttk.Label(frame, text="Min Pts:").grid(row=1, column=2, sticky="w", padx=(10,0), pady=(5,0))
        self.min_pts_entry = ttk.Entry(frame, textvariable=self.min_pts, state="disabled", width=8)
        self.min_pts_entry.grid(row=1, column=3, padx=5, sticky="ew", pady=(5,0))
        CreateToolTip(self.min_pts_entry, "Minimum number of points required to form a dense region (DBSCAN).")

    def _create_options_widgets(self, parent):
        frame = self._create_frame(parent, "Options")
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="Crop Side (px):").grid(row=0, column=0, sticky="w")
        self.crop_entry = ttk.Entry(frame, textvariable=self.crop_size, width=8)
        self.crop_entry.grid(row=0, column=1, sticky="ew", padx=5)
        CreateToolTip(self.crop_entry, "The side length for a square crop of each detected event.")

        self.subtract_mean_checkbox = ttk.Checkbutton(frame, text="Mean Subtraction", variable=self.subtract_mean_var)
        self.subtract_mean_checkbox.grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        CreateToolTip(self.subtract_mean_checkbox, "Subtract the calculated mean from each image.")

    def _create_method_widgets(self, parent):
        frame = self._create_frame(parent, "Method")
        self.method.trace_add("write", lambda *args: self.on_method_change())
        common_args = {"variable": self.method, "style": "Toolbutton"}
        
        self.dbscan_radio = ttk.Radiobutton(frame, text="DBSCAN", value="DBSCAN", **common_args)
        self.dbscan_radio.pack(side="left", expand=True, fill="x")
        CreateToolTip(self.dbscan_radio, "Detect and categorize events as a cluster on the basis of the DBSCAN method.")

        self.log_radio = ttk.Radiobutton(frame, text="LoG", value="LoG", **common_args)
        self.log_radio.pack(side="left", expand=True, fill="x")
        CreateToolTip(self.log_radio, "LoG detection (Experimental): Uses the convolution of the image with a Laplacian of Gaussian kernel for detection of events.")

    def _create_stacking_widgets(self, parent):
        frame = self._create_frame(parent, "Stacking")
        frame.columnconfigure(1, weight=1)

        stack_check = ttk.Checkbutton(frame, text="Stack Frames", variable=self.stack_var, command=self.stack_method, style="Switch.TCheckbutton")
        stack_check.grid(row=0, column=0, padx=5)
        CreateToolTip(stack_check, "(Experimental) Stack the frames, creating a resulting image presenting the overlay of all detections.")

    def _create_process_widgets(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(side="top", fill="x", padx=5, pady=10, expand=True)
        frame.columnconfigure((0, 1, 2), weight=1)
        self.start_button = ttk.Button(frame, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=2)
        self.cancel_button = ttk.Button(frame, text="Cancel Processing", state="disabled", command=self.cancel_processing)
        self.cancel_button.grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(frame, text="Open Output", command=self.open_output_folder).grid(row=0, column=2, sticky="ew", padx=2)

    def _create_display_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="Display", padding=5)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.back.set_viewer_frame(frame)
        
        fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(False)
        self.canvas = FigureCanvasTkAgg(fig, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        cmap_frame = ttk.Frame(frame)
        cmap_frame.grid(row=1, column=0, sticky="ew", pady=5)
        ttk.Label(cmap_frame, text="Colormap:").pack(side="left", padx=5)
        ttk.Combobox(cmap_frame, textvariable=self.colormap_selected, values=sorted(plt.colormaps())).pack(side="left", expand=True, fill="x", padx=5)

    def _create_console_widgets(self, parent):
        frame = ttk.LabelFrame(parent, text="Output", padding=10)
        frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        
        self.console_text = tk.Text(frame, wrap="word", height=5, state="normal", bg="#313131", fg="white", relief="flat")
        self.console_text.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.console_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.console_text.config(yscrollcommand=scrollbar.set)
    
    def stack_method(self):
        is_stacking = self.stack_var.get()

        if is_stacking:
            self.method.set("stack")

            for widget in [
                self.dbscan_radio,
                self.log_radio,
                self.min_pts_entry,
                self.subtract_mean_checkbox,
                self.k_value_entry,
                self.eps_entry,
                self.sigmaentry,
                self.crop_entry
            ]:
                widget.config(state="disabled")

        else:
            for widget in [
                self.dbscan_radio,
                self.log_radio,
                self.min_pts_entry,
                self.subtract_mean_checkbox,
                self.k_value_entry,
                self.eps_entry,
                self.sigmaentry,
                self.crop_entry
            ]:
                widget.config(state="normal")

            self.method.set("DBSCAN")
            self.on_method_change()



    def on_method_change(self):
        method = self.method.get()


        widgets = [
            self.k_value_entry,
            self.eps_entry,
            self.min_pts_entry,
            self.sigmaentry,
            self.subtract_mean_checkbox,
            self.crop_entry
        ]

        for w in widgets:
            w.config(state="disabled")


        if method == "DBSCAN":
            self.k_value_entry.config(state="normal")
            self.eps_entry.config(state="normal")
            self.min_pts_entry.config(state="normal")
            self.subtract_mean_checkbox.config(state="normal")
            self.crop_entry.config(state="normal")

        elif method == "LoG":
            self.sigmaentry.config(state="normal")
            self.crop_entry.config(state="normal")

    
    def process_images(self):
        try:
            input_dir, output_dir = self.input_dir.get(), self.output_dir.get()
            file_paths = glob.glob(os.path.join(input_dir, '*'))
            if not file_paths: 
                self.console_log("No RAW files found.")
                return
            
            self.cancel_event.clear()
            matrix_threshold, mean_matrix = self.back.threshold(file_paths, self.k_value.get(), output_dir)
            
            if matrix_threshold is None: 
                self.console_log("Processing cancelled."); 
                return
            
            cluster_sizes_all = []
            for file_path in file_paths:
                if self.cancel_event.is_set(): 
                    return
                with rawpy.imread(file_path) as raw:
                    raw_image = raw.raw_image_visible.copy().astype(np.float32)
                
                indices = np.argwhere(raw_image > matrix_threshold)
                clusters, sizes = self.back.clusteringDBSCAN(indices, self.epsilon.get(), self.min_pts.get())
                
                cluster_sizes_all.extend(sizes)
                image_to_crop = np.maximum(raw_image - mean_matrix, 0) if self.subtract_mean_var.get() else raw_image
                self.back.crop_n_save(os.path.basename(file_path).split('.')[0], image_to_crop, clusters, output_dir, self.colormap_selected.get(), self.crop_size.get())
            
            if not self.cancel_event.is_set():
                self.back.plot_histogram(cluster_sizes_all, output_dir)
                self.console_log("Processing completed.")
        except Exception as e: 
            self.console_log(f"Error: {e}")
        finally: 
            self._processing_done()
    
    def process_image_LoG(self):
        try:
            output_dir = self.output_dir.get()
            file_paths = glob.glob(os.path.join(self.input_dir.get(), '*'))
            if not file_paths: 
                self.console_log("No RAW files found.")
                return

            self.cancel_event.clear()
            detections_all = 0
            for file_path in file_paths:
                if self.cancel_event.is_set(): 
                    return
                with rawpy.imread(file_path) as raw:
                    raw_image = raw.raw_image_visible.copy().astype(np.float32)
                labeled_image = label(gaussian_laplace(raw_image, self.sigma.get()) < 0, connectivity=2)
                event_positions = [tuple(map(int, prop.centroid)) for prop in regionprops(labeled_image)]
                self.back.crop_n_save_LoG(os.path.basename(file_path).split('.')[0], raw_image, event_positions, output_dir, self.colormap_selected.get(), self.crop_size.get())
                detections_all += len(event_positions)
            if not self.cancel_event.is_set(): self.console_log(f"{detections_all} events detected in total.")
        except Exception as e: self.console_log(f"An error occurred: {e}")
        finally: self._processing_done()


    def stack_images(self):
        try:
            output_dir = self.output_dir.get()
            file_paths = glob.glob(os.path.join(self.input_dir.get(), '*'))
            mean_matrix, _ = self.back.load_matrices(output_dir)
            if mean_matrix is None:
                self.console_log("Run a standard processing method first.")
                return

            self.cancel_event.clear()
            self.console_log("Stacking images...")
            n = len(file_paths)
            with rawpy.imread(file_paths[0]) as raw:
                sum_matrix = np.zeros_like(raw.raw_image_visible, dtype=np.float32)

            for idx, file_path in enumerate(file_paths):
                if self.cancel_event.is_set(): return
                with rawpy.imread(file_path) as raw:
                    frame = raw.raw_image_visible.astype(np.float32)
                    diff = frame - mean_matrix
                    diff[diff < 0] = 0
                    sum_matrix += diff
                self.console_log(f"Stacking image {idx + 1}/{n}.")

            stacked_image = sum_matrix
            stacked_image = np.log1p(stacked_image)

            dtype = np.uint16 if stacked_image.max() > 255 else np.uint8
            stacked_image_final = cv2.normalize(stacked_image, None, 0, np.iinfo(dtype).max, cv2.NORM_MINMAX).astype(dtype)
            cv2.imwrite(os.path.join(output_dir, "stacked_image.png"), stacked_image_final)
            self.update_image_viewer(stacked_image_final)
            self.console_log("Stacked image saved.")
        except Exception as e:
            self.console_log(f"An error occurred: {e}")
        finally:
            self._processing_done()



    def start_processing(self):
        if self.processing:
            return
        if not self.input_dir.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Please select both input and output directories.")
            return

        self.processing = True
        self.start_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.console_log("Processing started...")

        target_map = {
            "DBSCAN": self.process_images,
            "LoG": self.process_image_LoG,
            "stack": self.stack_images
        }

        self.worker_thread = threading.Thread(
            target=target_map[self.method.get()],
            daemon=True
        )
        self.worker_thread.start()


    def cancel_processing(self):
        if self.processing:
            self.cancel_event.set()
            
    def _processing_done(self, cancelled=False):
        self.processing = False
        self.start_button.config(state="normal")
        self.cancel_button.config(state="disabled")
        if cancelled: 
            self.console_log("Processing canceled.")
        
    def on_close(self):
        if self.processing: 
            self.cancel_processing()
        self.root.destroy()

    def browse_input(self):
        directory = filedialog.askdirectory()
        if directory: 
            self.input_dir.set(directory)

    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory: 
            self.output_dir.set(directory)

    def open_output_folder(self):
        output_dir = self.output_dir.get()
        if not output_dir or not os.path.exists(output_dir):
            messagebox.showerror("Error", "Output directory does not exist."); 
            return
        os.startfile(output_dir)


    def console_log(self, message):
        self.console_text.insert(tk.END, f"{message}\n")
        self.console_text.see(tk.END)

    def update_image_viewer(self, image):
        self.back.clear_histogram()
        self.ax.clear()
        self.ax.imshow(image, cmap=self.colormap_selected.get())
        self.canvas.draw()
    
class CreateToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.waittime = 500
        self.wraplength = 200
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
        self.unschedule(); 
        self.hidetip()
    def schedule(self): 
        self.unschedule(); 
        self.id = self.widget.after(self.waittime, self.showtip)
    def unschedule(self): 
        id = self.id; 
        self.id = None; 
        if id: 
            self.widget.after_cancel(id)
    def showtip(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(self.tw, text=self.text, justify='left', background="#313131", foreground="white", relief='solid', borderwidth=1, wraplength = self.wraplength)
        label.pack(ipadx=1, ipady=1)
    def hidetip(self): 
        tw = self.tw; 
        self.tw= None; 
        if tw: 
            tw.destroy()

if __name__ == "__main__":
    root = tk.Tk()

    base = os.path.dirname(os.path.abspath(__file__))
    theme_path = os.path.join(base, "Theme", "forest-dark.tcl")

    root.tk.call("source", theme_path)

    style = ttk.Style(root)
    style.theme_use("forest-dark")


    app = CMOSdec(root)
    root.mainloop()
