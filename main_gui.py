import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk  # Themed Tkinter for better look
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import shutil
import zipfile
import threading
from PIL import Image, ImageTk
import queue # For thread communication

# --- Default Parameters (can be overridden by UI) ---
DEFAULT_DILATION_KERNEL_WIDTH = 10 # Default UI value
DEFAULT_DILATION_KERNEL_HEIGHT = 10 # Default UI value
DEFAULT_DILATION_ITERATIONS = 3    # Default UI value

# --- Fixed Parameters (Could be added to UI later if needed) ---
THRESH_BLOCK_SIZE = 11
THRESH_C_SUBTRACT = 7
MIN_CONTOUR_AREA_RATIO = 0.0005
MAX_CONTOUR_AREA_RATIO = 0.1
PADDING = 10 # Increased default padding slightly
OUTPUT_DIR_BASE = 'extracted_words_temp' # Use a temp dir
# -----------------------------------------

# Global variable to store the PhotoImage object to prevent garbage collection
displayed_photo = None
original_img_for_display = None # Store the loaded CV2 image globally for resizing

def sort_contours_arabic(cnts):
    """Sorts contours top-to-bottom, then right-to-left."""
    if not cnts:
        return [], []
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    # Sort by y-coordinate first, then by x-coordinate in descending order (right-to-left)
    # Added a check in case cnts is empty
    try:
        (cnts_sorted, boundingBoxes_sorted) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: (b[1][1], -b[1][0]), reverse=False))
    except ValueError: # Handles case where zip has nothing to unpack
         return [], []
    return cnts_sorted, boundingBoxes_sorted

def update_status(message, clear=False):
    """Updates the status text area in the GUI."""
    status_text.config(state=tk.NORMAL)
    if clear:
        status_text.delete('1.0', tk.END)
    status_text.insert(tk.END, message + "\n")
    status_text.see(tk.END) # Scroll to the end
    status_text.config(state=tk.DISABLED)
    # Force GUI update without blocking excessively
    root.update_idletasks()


def display_image_in_gui(img_cv2, max_width=500, max_height=400):
    """Displays an OpenCV image (BGR) in the Tkinter image panel."""
    global displayed_photo # Keep a reference to avoid garbage collection

    if img_cv2 is None:
        image_panel.config(image=None)
        displayed_photo = None
        return

    # Resize image to fit panel while maintaining aspect ratio
    h, w = img_cv2.shape[:2]
    scale = min(max_width/w, max_height/h)
    if scale < 1: # Only scale down
        new_w, new_h = int(w*scale), int(h*scale)
        img_resized = cv2.resize(img_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_cv2 # Use original if it fits

    # Convert BGR (OpenCV) to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # Convert NumPy array to PIL Image
    img_pil = Image.fromarray(img_rgb)
    # Convert PIL Image to Tkinter PhotoImage
    displayed_photo = ImageTk.PhotoImage(img_pil)

    # Update the label
    image_panel.config(image=displayed_photo)
    image_panel.image = displayed_photo # Keep reference


def extract_words_threaded(image_path, dilation_kernel_w, dilation_kernel_h, dilation_iterations, result_queue):
    """
    Worker function to run extraction in a separate thread.
    Uses a queue to communicate results back to the main thread.
    """
    try:
        update_status(f"Processing image: {os.path.basename(image_path)}", clear=True)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error: Could not read image file {image_path}")

        img_height, img_width = img.shape[:2]
        total_image_area = img_height * img_width
        update_status(f"Image dimensions: {img_width}x{img_height}")
        global original_img_for_display
        original_img_for_display = img.copy() # Store for potential resizing display

        # 1. Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, THRESH_BLOCK_SIZE, THRESH_C_SUBTRACT)
        update_status("Applied adaptive thresholding.")

        # 2. Morphological Operations (Dilation)
        dilation_kernel_size = (dilation_kernel_w, dilation_kernel_h)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel_size)
        dilated = cv2.dilate(binary_inv, kernel, iterations=dilation_iterations)
        update_status(f"Applied dilation with kernel {dilation_kernel_size} and {dilation_iterations} iterations.")

        # 3. Find Contours
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        update_status(f"Found {len(contours)} initial contours.")

        # 4. Filter Contours
        valid_contours = []
        min_area = total_image_area * MIN_CONTOUR_AREA_RATIO
        max_area = total_image_area * MAX_CONTOUR_AREA_RATIO
        update_status(f"Filtering contours: Min Area = {min_area:.2f}, Max Area = {max_area:.2f}")

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                valid_contours.append(cnt)

        update_status(f"Found {len(valid_contours)} valid contours after filtering.")

        if not valid_contours:
            update_status("No valid word contours found. Try adjusting parameters.")
            result_queue.put({"success": True, "count": 0, "output_dir": None, "image_with_boxes": img})
            return

        # 5. Sort Contours
        contours_sorted, boundingBoxes_sorted = sort_contours_arabic(valid_contours)

        # Prepare output directory
        output_dir = OUTPUT_DIR_BASE
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        update_status(f"Created temporary output directory: {output_dir}")

        img_with_boxes = img.copy()

        # 6. Crop and Save
        word_count = 0
        for i, (cnt, bbox) in enumerate(zip(contours_sorted, boundingBoxes_sorted)):
            x, y, w, h = bbox
            word_count += 1

            y_start = max(0, y - PADDING)
            y_end = min(img_height, y + h + PADDING)
            x_start = max(0, x - PADDING)
            x_end = min(img_width, x + w + PADDING)

            cropped_word = img[y_start:y_end, x_start:x_end]

            output_filename = os.path.join(output_dir, f"{word_count}.png")
            try:
                cv2.imwrite(output_filename, cropped_word)
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green box
                cv2.putText(img_with_boxes, str(word_count), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Red number
            except Exception as e:
                update_status(f"Warning: Error saving file {output_filename}: {e}")
                word_count -= 1

        update_status(f"\nSuccessfully extracted {word_count} word images.")
        update_status("Displaying image with detected word boxes (Red numbers indicate order).")

        result_queue.put({
            "success": True,
            "count": word_count,
            "output_dir": output_dir,
            "image_with_boxes": img_with_boxes
        })

    except Exception as e:
        # Put error details in the queue
        import traceback
        error_msg = f"An error occurred:\n{traceback.format_exc()}"
        result_queue.put({"success": False, "error": error_msg})


# --- GUI Functions ---

def select_image():
    """Opens a file dialog to select an image."""
    global original_img_for_display
    filepath = filedialog.askopenfilename(
        title="Select Handwritten Arabic Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All Files", "*.*")]
    )
    if filepath:
        input_filepath.set(filepath)
        update_status(f"Selected image: {filepath}", clear=True)
        # Display the selected image immediately
        try:
            img = cv2.imread(filepath)
            if img is None:
                 messagebox.showerror("Error", f"Could not read image file:\n{filepath}")
                 return
            original_img_for_display = img.copy() # Store for resizing later
            display_image_in_gui(img)
        except Exception as e:
             messagebox.showerror("Error", f"Failed to load or display image:\n{e}")
             display_image_in_gui(None) # Clear panel on error
    else:
        input_filepath.set("")
        update_status("Image selection cancelled.", clear=True)
        display_image_in_gui(None)
        original_img_for_display = None

def start_extraction():
    """Starts the word extraction process in a separate thread."""
    image_path = input_filepath.get()
    if not image_path or not os.path.exists(image_path):
        messagebox.showerror("Error", "Please select a valid image file first.")
        return

    try:
        # Get parameters from UI, validating they are integers
        dilation_w = int(dilation_width_var.get())
        dilation_h = int(dilation_height_var.get())
        iterations = int(dilation_iterations_var.get())

        if dilation_w <= 0 or dilation_h <= 0 or iterations <= 0:
             raise ValueError("Dilation dimensions and iterations must be positive integers.")

    except ValueError as e:
        messagebox.showerror("Invalid Input", f"Please enter valid positive integers for dilation parameters.\nError: {e}")
        return

    # Disable button, clear status, show progress indicator
    extract_button.config(state=tk.DISABLED)
    save_button.config(state=tk.DISABLED) # Disable save until results are ready
    update_status("Starting extraction...", clear=True)
    # Could add a progress bar here if desired

    # Clear previous result image
    # display_image_in_gui(None) # Keep original image displayed during processing

    # Create a queue for thread communication
    result_queue = queue.Queue()

    # Start the worker thread
    extraction_thread = threading.Thread(
        target=extract_words_threaded,
        args=(image_path, dilation_w, dilation_h, iterations, result_queue),
        daemon=True # Allows main program to exit even if thread is running
    )
    extraction_thread.start()

    # Schedule a check for the result queue
    root.after(100, check_extraction_queue, result_queue)

def check_extraction_queue(result_queue):
    """Checks the queue for results from the worker thread."""
    global last_output_dir # Store the output dir path for saving
    try:
        result = result_queue.get_nowait() # Check without blocking

        # Re-enable button
        extract_button.config(state=tk.NORMAL)

        if result["success"]:
            word_count = result["count"]
            last_output_dir = result["output_dir"]
            image_with_boxes = result["image_with_boxes"]

            # Display the result image
            display_image_in_gui(image_with_boxes)

            if word_count > 0 and last_output_dir:
                update_status(f"Extraction complete. Found {word_count} words.")
                update_status(f"Temporary files saved in: {last_output_dir}")
                update_status("Click 'Save Results' to create a zip file.")
                save_button.config(state=tk.NORMAL) # Enable save button
            elif word_count == 0:
                 update_status("Extraction complete. No words found.")
                 save_button.config(state=tk.DISABLED)
            else: # Success but no output dir? Should not happen with current logic
                 update_status("Extraction finished, but no output directory generated.")
                 save_button.config(state=tk.DISABLED)

        else:
            # Error occurred in thread
            error_message = result.get("error", "Unknown error occurred in extraction thread.")
            messagebox.showerror("Extraction Error", error_message)
            update_status("Extraction failed.", clear=True)
            update_status(error_message)
            display_image_in_gui(None) # Clear image panel on error
            save_button.config(state=tk.DISABLED)
            last_output_dir = None

    except queue.Empty:
        # Queue is empty, means thread is still running
        # Reschedule the check
        root.after(100, check_extraction_queue, result_queue)


def save_results():
    """Zips the extracted words and asks the user where to save the zip file."""
    if not last_output_dir or not os.path.isdir(last_output_dir):
        messagebox.showerror("Error", "No valid extraction results found to save.")
        return

    # Suggest a filename based on the input image name
    input_file = input_filepath.get()
    if input_file:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        suggested_filename = f"{base_name}_words.zip"
    else:
        suggested_filename = "extracted_words.zip"

    zip_filepath = filedialog.asksaveasfilename(
        title="Save Extracted Words As Zip",
        defaultextension=".zip",
        initialfile=suggested_filename,
        filetypes=[("Zip Files", "*.zip")]
    )

    if not zip_filepath:
        update_status("Save cancelled.")
        return

    update_status(f"Zipping files into: {zip_filepath}")
    try:
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files_in_dir in os.walk(last_output_dir):
                for file in files_in_dir:
                    file_path = os.path.join(root, file)
                    # Arcname ensures files are directly in the zip, not inside OUTPUT_DIR_BASE folder
                    arcname = os.path.relpath(file_path, last_output_dir)
                    zipf.write(file_path, arcname=arcname)

        update_status(f"Successfully saved zip file: {zip_filepath}")
        messagebox.showinfo("Success", f"Extracted words saved to:\n{zip_filepath}")

        # Optional: Clean up temporary directory after successful zip
        # try:
        #     shutil.rmtree(last_output_dir)
        #     update_status(f"Cleaned up temporary directory: {last_output_dir}")
        #     last_output_dir = None # Clear the path after cleanup
        #     save_button.config(state=tk.DISABLED) # Disable save after cleanup
        # except Exception as e:
        #     update_status(f"Warning: Could not remove temporary directory {last_output_dir}: {e}")

    except Exception as e:
        import traceback
        error_msg = f"Error creating zip file:\n{traceback.format_exc()}"
        messagebox.showerror("Zip Error", error_msg)
        update_status(f"Failed to create zip file. Error: {e}")

def on_resize(event):
     """Handles window resize to redraw the image."""
     # Check if we have an image stored to redisplay
     if original_img_for_display is not None:
          # Use the panel's current size as max constraints
          display_image_in_gui(original_img_for_display, event.width, event.height)
     else:
          # If no image is loaded, ensure the panel is cleared or shows placeholder
          display_image_in_gui(None)

# --- GUI Setup ---
root = tk.Tk()
root.title("Handwritten Arabic Word Extractor")
# root.geometry("800x700") # Optional: Set initial size

# Variables
input_filepath = tk.StringVar()
dilation_width_var = tk.StringVar(value=str(DEFAULT_DILATION_KERNEL_WIDTH))
dilation_height_var = tk.StringVar(value=str(DEFAULT_DILATION_KERNEL_HEIGHT))
dilation_iterations_var = tk.StringVar(value=str(DEFAULT_DILATION_ITERATIONS))
last_output_dir = None # To store the path of the latest results

# --- Style ---
style = ttk.Style()
# Use a theme that looks better across platforms if available
available_themes = style.theme_names()
if 'clam' in available_themes:
    style.theme_use('clam')
elif 'vista' in available_themes: # Good fallback for Windows
    style.theme_use('vista')
# Add some padding to widgets
style.configure("TLabel", padding=5)
style.configure("TButton", padding=5)
style.configure("TEntry", padding=(5, 5, 5, 5)) # Left, Top, Right, Bottom
style.configure("TFrame", padding=10)

# --- Main Frame ---
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Make the main frame grid responsive
main_frame.columnconfigure(0, weight=1) # Input/Params column
main_frame.columnconfigure(1, weight=3) # Image/Status column
main_frame.rowconfigure(2, weight=1) # Image row
main_frame.rowconfigure(4, weight=1) # Status text row


# --- Left Column: Input and Controls ---
controls_frame = ttk.Frame(main_frame)
controls_frame.grid(row=0, column=0, rowspan=5, sticky="nsew", padx=10, pady=10)
controls_frame.columnconfigure(1, weight=1) # Make entry widgets expand


# File Selection
ttk.Label(controls_frame, text="Image File:").grid(row=0, column=0, sticky="w", pady=(0, 5))
entry_filepath = ttk.Entry(controls_frame, textvariable=input_filepath, state="readonly", width=40)
entry_filepath.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 5))
select_button = ttk.Button(controls_frame, text="Select Image...", command=select_image)
select_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 15))

# Dilation Parameters
ttk.Label(controls_frame, text="--- Dilation Parameters ---").grid(row=3, column=0, columnspan=2, pady=(10, 5), sticky="w")

ttk.Label(controls_frame, text="Kernel Width:").grid(row=4, column=0, sticky="w", pady=2)
entry_dilation_w = ttk.Entry(controls_frame, textvariable=dilation_width_var, width=10)
entry_dilation_w.grid(row=4, column=1, sticky="ew", pady=2)

ttk.Label(controls_frame, text="Kernel Height:").grid(row=5, column=0, sticky="w", pady=2)
entry_dilation_h = ttk.Entry(controls_frame, textvariable=dilation_height_var, width=10)
entry_dilation_h.grid(row=5, column=1, sticky="ew", pady=2)

ttk.Label(controls_frame, text="Iterations:").grid(row=6, column=0, sticky="w", pady=2)
entry_dilation_iter = ttk.Entry(controls_frame, textvariable=dilation_iterations_var, width=10)
entry_dilation_iter.grid(row=6, column=1, sticky="ew", pady=2)


# Action Buttons
extract_button = ttk.Button(controls_frame, text="Extract Words", command=start_extraction)
extract_button.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(20, 5))

save_button = ttk.Button(controls_frame, text="Save Results...", command=save_results, state=tk.DISABLED)
save_button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(5, 10))


# --- Right Column: Image Display and Status ---
# Frame for Image Display (helps with resizing)
image_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=1)
image_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=(0,10), pady=(0, 5))
image_frame.columnconfigure(0, weight=1)
image_frame.rowconfigure(0, weight=1)

# Image Panel (using Label)
image_panel = ttk.Label(image_frame, text="Select an image to display", anchor="center")
image_panel.grid(row=0, column=0, sticky="nsew")
# Bind resize event to the frame containing the label
image_frame.bind("<Configure>", on_resize)


# Status Area Label
ttk.Label(main_frame, text="Status Log:").grid(row=3, column=1, sticky="sw", padx=(0,10))

# Status Text Area
status_text = scrolledtext.ScrolledText(main_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
status_text.grid(row=4, column=1, sticky="nsew", padx=(0,10), pady=(0,10))


# Start the Tkinter main loop
root.mainloop()

# Cleanup temporary directory on exit if it still exists
if last_output_dir and os.path.exists(last_output_dir):
     try:
          print(f"Cleaning up temporary directory: {last_output_dir}")
          shutil.rmtree(last_output_dir)
     except Exception as e:
          print(f"Warning: Could not remove temporary directory {last_output_dir} on exit: {e}")
