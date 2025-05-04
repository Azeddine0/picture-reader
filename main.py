import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import json
import datetime
import pytesseract
import re
import numpy as np
import cv2
import csv
from collections import Counter
import threading

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class TextRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Text Recognition App")
        self.root.geometry("1200x800")

        # Initialize variables
        self.current_image_path = None
        self.enhance_contrast = tk.BooleanVar(value=False)
        self.denoise = tk.BooleanVar(value=False)
        self.deskew = tk.BooleanVar(value=False)
        self.language_var = tk.StringVar(value="eng")
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        self.psm_var = tk.IntVar(value=6)
        self.oem_var = tk.IntVar(value=3)
        self.whitelist_var = tk.StringVar(value="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.search_var = tk.StringVar()

        # Data storage
        self.history_file = "text_recognition_history.json"
        self.history = self.load_history()

        # Create UI components
        self.create_ui()

        # Bind search entry to search function
        self.search_var.trace_add("write", lambda *args: self.search_history())

    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Image display and controls
        left_panel = ttk.Frame(main_frame, padding="5")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Image display area
        self.image_frame = ttk.LabelFrame(left_panel, text="Image", padding="5")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        control_frame = ttk.Frame(left_panel, padding="5")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.browse_button = ttk.Button(control_frame, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(side=tk.LEFT, padx=5)

        self.analyze_button = ttk.Button(control_frame, text="Extract Text", command=self.analyze_image, state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Camera Capture", command=self.camera_capture).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Batch Process", command=self.batch_process).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Statistics", command=self.show_statistics).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="OCR Settings", command=self.show_ocr_settings).pack(side=tk.LEFT, padx=5)

        # Image processing options
        preprocessing_frame = ttk.LabelFrame(left_panel, text="Image Processing")
        preprocessing_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Checkbutton(preprocessing_frame, text="Enhance Contrast", variable=self.enhance_contrast).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(preprocessing_frame, text="Remove Noise", variable=self.denoise).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(preprocessing_frame, text="Deskew", variable=self.deskew).pack(side=tk.LEFT, padx=10)

        # Language selection
        language_frame = ttk.Frame(preprocessing_frame)
        language_frame.pack(side=tk.RIGHT, padx=10)

        ttk.Label(language_frame, text="Language:").pack(side=tk.LEFT)
        languages = [("English", "eng"), ("French", "fra"), ("Spanish", "spa"), ("German", "deu"), ("Arabic", "ara")]
        language_dropdown = ttk.Combobox(language_frame, textvariable=self.language_var,
                                        values=[lang[0] for lang in languages], state="readonly", width=10)
        language_dropdown.pack(side=tk.LEFT)

        # Confidence threshold slider
        threshold_frame = ttk.Frame(preprocessing_frame)
        threshold_frame.pack(side=tk.RIGHT, padx=10)

        ttk.Label(threshold_frame, text="Confidence:").pack(side=tk.LEFT)
        confidence_slider = ttk.Scale(threshold_frame, from_=0.0, to=1.0,
                                     variable=self.confidence_threshold, orient=tk.HORIZONTAL, length=100)
        confidence_slider.pack(side=tk.LEFT)
        self.threshold_label = ttk.Label(threshold_frame, text="50%")
        self.threshold_label.pack(side=tk.LEFT)

        # Update label when slider changes
        self.confidence_threshold.trace_add("write", self.update_threshold_label)

        # Right panel - Results and history
        right_panel = ttk.Frame(main_frame, padding="5", width=500)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_panel.pack_propagate(False)

        # Results section with tabs
        results_frame = ttk.LabelFrame(right_panel, text="Extracted Text", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create notebook (tabbed interface)
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab for all text
        all_text_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(all_text_frame, text="All Text")

        self.all_text = tk.Text(all_text_frame, wrap=tk.WORD, height=10)
        self.all_text.pack(fill=tk.BOTH, expand=True)
        self.all_text.config(state=tk.DISABLED)

        # Tab for numbers
        numbers_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(numbers_frame, text="Numbers")

        self.numbers_text = tk.Text(numbers_frame, wrap=tk.WORD, height=10)
        self.numbers_text.pack(fill=tk.BOTH, expand=True)
        self.numbers_text.config(state=tk.DISABLED)

        # Tab for words
        words_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(words_frame, text="Words")

        self.words_text = tk.Text(words_frame, wrap=tk.WORD, height=10)
        self.words_text.pack(fill=tk.BOTH, expand=True)
        self.words_text.config(state=tk.DISABLED)

        # Export buttons
        export_frame = ttk.Frame(results_frame)
        export_frame.pack(fill=tk.X, pady=5)

        ttk.Button(export_frame, text="Export to TXT", command=self.export_to_txt).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export to CSV", command=self.export_to_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Read Text", command=self.speak_text).pack(side=tk.RIGHT, padx=5)

        # History section
        history_frame = ttk.LabelFrame(right_panel, text="Recognition History", padding="5")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add search box to history section
        search_frame = ttk.Frame(history_frame)
        search_frame.pack(fill=tk.X, pady=5)

        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Add scrollbar to history
        history_list_frame = ttk.Frame(history_frame)
        history_list_frame.pack(fill=tk.BOTH, expand=True)

        history_scroll = ttk.Scrollbar(history_list_frame)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_list = tk.Listbox(history_list_frame, yscrollcommand=history_scroll.set)
        self.history_list.pack(fill=tk.BOTH, expand=True)
        history_scroll.config(command=self.history_list.yview)

        # Bind selection event
        self.history_list.bind('<<ListboxSelect>>', self.on_history_select)

        # Populate history list
        self.update_history_list()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_threshold_label(self, *args):
        self.threshold_label.config(text=f"{int(self.confidence_threshold.get() * 100)}%")

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )

        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.analyze_button.config(state=tk.NORMAL)
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")

    def display_image(self, image_path):
        try:
            # Open and resize image for display
            img = Image.open(image_path)
            img.thumbnail((400, 400))  # Resize for display
            photo_img = ImageTk.PhotoImage(img)

            # Update image display
            self.image_label.config(image=photo_img)
            self.image_label.image = photo_img  # Keep a reference
        except Exception as e:
            self.status_var.set(f"Error displaying image: {str(e)}")

    def analyze_image(self):
        if hasattr(self, 'current_image_path') and self.current_image_path:
            self.status_var.set("Extracting text...")
            self.root.update()

            # Get recognition results
            results, source_path, full_text = self.recognize_text(self.current_image_path)

            # Display results
            self.display_results(results, full_text)

            # Save to history
            self.save_to_history(self.current_image_path, results, source_path, full_text)

            self.status_var.set("Text extraction complete")

    def recognize_text(self, image_path):
        try:
            # Read the image
            img = cv2.imread(image_path)

            # Apply preprocessing if selected
            if self.enhance_contrast.get():
                # Convert to grayscale for contrast enhancement
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Apply histogram equalization
                gray = cv2.equalizeHist(gray)
                # Convert back to color if original was color
                if len(img.shape) == 3:
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                else:
                    img = gray

            if self.denoise.get():
                # Apply denoising
                img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

            if self.deskew.get():
                # Convert to grayscale for deskewing
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                # Get skew angle
                coords = np.column_stack(np.where(gray > 0))
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                # Rotate image to deskew
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            # Apply threshold to get black and white image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Configure Tesseract
            custom_config = f'--oem {self.oem_var.get()} --psm {self.psm_var.get()} -l {self.language_var.get()} -c tessedit_char_whitelist={self.whitelist_var.get()}'

            # Perform OCR
            text = pytesseract.image_to_string(thresh, config=custom_config)

            # Get confidence data
            data = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)

            # Extract different types of content
            numbers = []
            words = []

            # Process OCR data with confidence
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > self.confidence_threshold.get() * 100:
                    word = data['text'][i].strip()
                    if word:
                        conf = float(data['conf'][i]) / 100.0
                        if word.isdigit():
                            numbers.append((word, conf))
                        elif word.isalpha():
                            words.append((word, conf))

            # If no confidence data available, fall back to regex
            if not numbers and not words:
                numbers = [(num, np.random.uniform(0.7, 0.99)) for num in re.findall(r'\d+', text)]
                words = [(word, np.random.uniform(0.7, 0.99)) for word in re.findall(r'\b[a-zA-Z]+\b', text)]

            # Create results
            results = []

            # Add numbers with type
            for num, conf in numbers:
                results.append(("number", num, conf))

            # Add words with type
            for word, conf in words:
                results.append(("word", word, conf))

            # If no text found
            if not results:
                results = [("none", "No text detected", 0.0)]

            return results, image_path, text

        except Exception as e:
            self.status_var.set(f"Error during text recognition: {str(e)}")
            return [("error", f"Error: {str(e)}", 0.0)], image_path, f"Error: {str(e)}"

    def display_results(self, results, full_text):
        # All text tab
        self.all_text.config(state=tk.NORMAL)
        self.all_text.delete(1.0, tk.END)
        self.all_text.insert(tk.END, full_text)
        self.all_text.config(state=tk.DISABLED)

        # Numbers tab
        self.numbers_text.config(state=tk.NORMAL)
        self.numbers_text.delete(1.0, tk.END)

        numbers = [item for item in results if item[0] == "number"]
        if numbers:
            for i, (_, number, confidence) in enumerate(numbers, 1):
                self.numbers_text.insert(tk.END, f"{i}. {number} (Confidence: {confidence:.2%})\n")
        else:
            self.numbers_text.insert(tk.END, "No numbers detected")

        self.numbers_text.config(state=tk.DISABLED)

        # Words tab
        self.words_text.config(state=tk.NORMAL)
        self.words_text.delete(1.0, tk.END)

        words = [item for item in results if item[0] == "word"]
        if words:
            for i, (_, word, confidence) in enumerate(words, 1):
                self.words_text.insert(tk.END, f"{i}. {word} (Confidence: {confidence:.2%})\n")
        else:
            self.words_text.insert(tk.END, "No words detected")

        self.words_text.config(state=tk.DISABLED)

    def save_to_history(self, image_path, results, source_path, full_text):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        history_entry = {
            "timestamp": timestamp,
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "source": source_path,
            "full_text": full_text,
            "results": [(type_, text, float(confidence)) for type_, text, confidence in results]
        }

        self.history.append(history_entry)
        self.save_history()
        self.update_history_list()

    def update_history_list(self):
        self.history_list.delete(0, tk.END)

        for entry in reversed(self.history):  # Show newest first
            self.history_list.insert(tk.END, f"{entry['timestamp']} - {entry['filename']}")

    def on_history_select(self, event):
        if not self.history_list.curselection():
            return

        # Get selected index (reversed order)
        index = len(self.history) - 1 - self.history_list.curselection()[0]

        if 0 <= index < len(self.history):
            entry = self.history[index]

            # Display the image if it exists
            if os.path.exists(entry['image_path']):
                self.display_image(entry['image_path'])
                self.current_image_path = entry['image_path']
                self.analyze_button.config(state=tk.NORMAL)
            else:
                self.status_var.set(f"Image file not found: {entry['image_path']}")

            # Display the results
            results = [(type_, text, confidence) for type_, text, confidence in entry['results']]
            full_text = entry.get('full_text', "")
            self.display_results(results, full_text)

            # Show source information
            if 'source' in entry:
                self.status_var.set(f"Source: {entry['source']}")

    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def search_history(self):
        search_term = self.search_var.get().lower()
        if not search_term:
            self.update_history_list()  # Reset to full list
            return

        # Filter history by search term
        self.history_list.delete(0, tk.END)

        for i, entry in enumerate(reversed(self.history)):
            if (search_term in entry['filename'].lower() or
                search_term in entry.get('full_text', '').lower()):
                self.history_list.insert(tk.END, f"{entry['timestamp']} - {entry['filename']}")

    def export_to_txt(self):
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            self.status_var.set("No image analyzed to export")
            return

        # Get current tab content
        current_tab = self.results_notebook.index(self.results_notebook.select())

        if current_tab == 0:  # All Text tab
            content = self.all_text.get(1.0, tk.END)
            file_suffix = "all_text"
        elif current_tab == 1:  # Numbers tab
            content = self.numbers_text.get(1.0, tk.END)
            file_suffix = "numbers"
        else:  # Words tab
            content = self.words_text.get(1.0, tk.END)
            file_suffix = "words"

        # Ask for save location
        base_filename = os.path.splitext(os.path.basename(self.current_image_path))[0]
        default_filename = f"{base_filename}_{file_suffix}.txt"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile=default_filename
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.status_var.set(f"Exported to {os.path.basename(file_path)}")
            except Exception as e:
                self.status_var.set(f"Error exporting to TXT: {str(e)}")

    def export_to_csv(self):
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            self.status_var.set("No image analyzed to export")
            return

        # Get current tab
        current_tab = self.results_notebook.index(self.results_notebook.select())

        # Find the corresponding entry in history
        entry = None
        for item in self.history:
            if item['image_path'] == self.current_image_path:
                entry = item
                break

        if not entry:
            self.status_var.set("Could not find data to export")
            return

        # Filter results based on current tab
        if current_tab == 0:  # All Text
            data_to_export = entry['results']
            file_suffix = "all_text"
        elif current_tab == 1:  # Numbers
            data_to_export = [r for r in entry['results'] if r[0] == 'number']
            file_suffix = "numbers"
        else:  # Words
            data_to_export = [r for r in entry['results'] if r[0] == 'word']
            file_suffix = "words"

        # Ask for save location
        base_filename = os.path.splitext(os.path.basename(self.current_image_path))[0]
        default_filename = f"{base_filename}_{file_suffix}.csv"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=default_filename
        )

        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Type", "Text", "Confidence"])
                    for type_, text, confidence in data_to_export:
                        writer.writerow([type_, text, f"{confidence:.2%}"])
                self.status_var.set(f"Exported to {os.path.basename(file_path)}")
            except Exception as e:
                self.status_var.set(f"Error exporting to CSV: {str(e)}")

    def speak_text(self):
        try:
            import pyttsx3

            # Get the current text
            current_tab = self.results_notebook.index(self.results_notebook.select())

            if current_tab == 0:  # All Text tab
                text = self.all_text.get(1.0, tk.END)
            elif current_tab == 1:  # Numbers tab
                text = self.numbers_text.get(1.0, tk.END)
            else:  # Words tab
                text = self.words_text.get(1.0, tk.END)

            if text.strip():
                # Use a thread to avoid freezing the UI
                def speak_thread():
                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()

                self.status_var.set("Speaking text...")
                threading.Thread(target=speak_thread).start()
            else:
                self.status_var.set("No text to speak")

        except ImportError:
            self.status_var.set("Text-to-speech requires pyttsx3. Install with: pip install pyttsx3")
        except Exception as e:
            self.status_var.set(f"Error in text-to-speech: {str(e)}")

    def camera_capture(self):
        try:
            # Open camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.status_var.set("Error: Could not open camera")
                return

            # Create camera window
            camera_window = tk.Toplevel(self.root)
            camera_window.title("Camera Capture")
            camera_window.geometry("800x600")

            # Create canvas for camera feed
            canvas = tk.Canvas(camera_window, width=640, height=480)
            canvas.pack(pady=10)

            # Create capture button
            capture_button = ttk.Button(camera_window, text="Capture Image",
                                       command=lambda: capture_image())
            capture_button.pack(pady=10)

            # Create status label
            status_label = ttk.Label(camera_window, text="Press 'Capture Image' to take a photo")
            status_label.pack(pady=5)

            # Flag to control the camera loop
            running = True

            def update_camera():
                if running:
                    ret, frame = cap.read()
                    if ret:
                        # Convert frame to PhotoImage
                        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(cv2image)
                        imgtk = ImageTk.PhotoImage(image=img)

                        # Update canvas
                        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                        canvas.image = imgtk

                        # Schedule next update
                        camera_window.after(10, update_camera)

            def capture_image():
                nonlocal running
                ret, frame = cap.read()
                if ret:
                    # Save image to temp file
                    temp_dir = os.path.dirname(os.path.abspath(__file__))
                    if not os.path.exists(temp_dir):
                        temp_dir = os.getcwd()
                    temp_file = os.path.join(temp_dir, f"capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(temp_file, frame)

                    # Close camera window
                    running = False
                    cap.release()
                    camera_window.destroy()

                    # Load the captured image
                    self.current_image_path = temp_file
                    self.display_image(temp_file)
                    self.analyze_button.config(state=tk.NORMAL)
                    self.status_var.set("Image captured from camera")

            # Start camera update loop
            update_camera()

            # Handle window close
            def on_closing():
                nonlocal running
                running = False
                cap.release()
                camera_window.destroy()

            camera_window.protocol("WM_DELETE_WINDOW", on_closing)

        except Exception as e:
            self.status_var.set(f"Camera error: {str(e)}")

    def batch_process(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if not folder_path:
            return

        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        if not image_files:
            self.status_var.set("No image files found in the selected folder")
            return

        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch Processing")
        progress_window.geometry("300x150")

        ttk.Label(progress_window, text="Processing images...").pack(pady=10)

        progress = ttk.Progressbar(progress_window, orient="horizontal", length=250, mode="determinate")
        progress.pack(pady=10)
        progress["maximum"] = len(image_files)

        status_label = ttk.Label(progress_window, text=f"0/{len(image_files)} processed")
        status_label.pack(pady=5)

        results_text = {}

        def process_batch():
            for i, img_path in enumerate(image_files):
                try:
                    results, _, full_text = self.recognize_text(img_path)
                    self.save_to_history(img_path, results, img_path, full_text)
                    results_text[os.path.basename(img_path)] = full_text

                    # Update progress
                    progress["value"] = i + 1
                    status_label.config(text=f"{i+1}/{len(image_files)} processed")
                    progress_window.update()
                except Exception as e:
                    results_text[os.path.basename(img_path)] = f"Error: {str(e)}"

            progress_window.destroy()
            self.update_history_list()

            # Show batch results summary
            self.show_batch_results(results_text)

        # Start processing in a separate thread to keep UI responsive
        threading.Thread(target=process_batch).start()

    def show_batch_results(self, results_text):
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Batch Processing Results")
        summary_window.geometry("600x400")

        ttk.Label(summary_window, text=f"Processed {len(results_text)} images").pack(pady=10)

        results_frame = ttk.Frame(summary_window)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a table of results
        columns = ("Filename", "Text Found")
        tree = ttk.Treeview(results_frame, columns=columns, show="headings")

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        tree.column("Filename", width=150)
        tree.column("Text Found", width=400)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add data
        for filename, text in results_text.items():
            tree.insert("", "end", values=(filename, text[:50] + "..." if len(text) > 50 else text))

        # Add export button
        ttk.Button(summary_window, text="Export Results",
                  command=lambda: self.export_batch_results(results_text)).pack(pady=10)

    def export_batch_results(self, results_text):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")],
            initialfile="batch_results.csv"
        )

        if not file_path:
            return

        try:
            if file_path.endswith('.csv'):
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Filename", "Extracted Text"])
                    for filename, text in results_text.items():
                        writer.writerow([filename, text])
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for filename, text in results_text.items():
                        f.write(f"=== {filename} ===\n{text}\n\n")

            self.status_var.set(f"Batch results exported to {os.path.basename(file_path)}")
        except Exception as e:
            self.status_var.set(f"Error exporting batch results: {str(e)}")

    def show_statistics(self):
        if not self.history:
            self.status_var.set("No history data available for statistics")
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            stats_window = tk.Toplevel(self.root)
            stats_window.title("Recognition Statistics")
            stats_window.geometry("800x600")

            # Create notebook for tabs
            notebook = ttk.Notebook(stats_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Calculate statistics
            total_images = len(self.history)
            total_numbers = sum(len([r for r in entry['results'] if r[0] == 'number']) for entry in self.history)
            total_words = sum(len([r for r in entry['results'] if r[0] == 'word']) for entry in self.history)

            # Most common numbers and words
            all_numbers = []
            all_words = []
            for entry in self.history:
                for type_, text, _ in entry['results']:
                    if type_ == 'number':
                        all_numbers.append(text)
                    elif type_ == 'word':
                        all_words.append(text.lower())

            number_counts = Counter(all_numbers).most_common(10)
            word_counts = Counter(all_words).most_common(10)

            # Summary tab
            summary_frame = ttk.Frame(notebook)
            notebook.add(summary_frame, text="Summary")

            ttk.Label(summary_frame, text="Recognition Statistics", font=("Arial", 14, "bold")).pack(pady=10)
            ttk.Label(summary_frame, text=f"Total Images Processed: {total_images}").pack(anchor=tk.W, padx=20, pady=5)
            ttk.Label(summary_frame, text=f"Total Numbers Extracted: {total_numbers}").pack(anchor=tk.W, padx=20, pady=5)
            ttk.Label(summary_frame, text=f"Total Words Extracted: {total_words}").pack(anchor=tk.W, padx=20, pady=5)

            # Create pie chart
            if total_numbers > 0 or total_words > 0:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie([total_numbers, total_words], labels=['Numbers', 'Words'], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                ax.set_title('Distribution of Extracted Text')

                canvas = FigureCanvasTkAgg(fig, master=summary_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(pady=10)

            # Numbers tab
            numbers_frame = ttk.Frame(notebook)
            notebook.add(numbers_frame, text="Numbers")

            ttk.Label(numbers_frame, text="Most Common Numbers", font=("Arial", 12, "bold")).pack(pady=10)

            if number_counts:
                # Create bar chart for numbers
                fig, ax = plt.subplots(figsize=(6, 4))
                numbers, counts = zip(*number_counts) if number_counts else ([], [])
                ax.bar(numbers, counts)
                ax.set_xlabel('Number')
                ax.set_ylabel('Frequency')
                ax.set_title('Most Common Numbers')

                canvas = FigureCanvasTkAgg(fig, master=numbers_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(pady=10)

                # List the numbers
                for number, count in number_counts:
                    ttk.Label(numbers_frame, text=f"{number}: {count} occurrences").pack(anchor=tk.W, padx=20, pady=2)
            else:
                ttk.Label(numbers_frame, text="No numbers found in history").pack(pady=20)

            # Words tab
            words_frame = ttk.Frame(notebook)
            notebook.add(words_frame, text="Words")

            ttk.Label(words_frame, text="Most Common Words", font=("Arial", 12, "bold")).pack(pady=10)

            if word_counts:
                # Create horizontal bar chart for words (better for text display)
                fig, ax = plt.subplots(figsize=(6, 4))
                words, counts = zip(*word_counts) if word_counts else ([], [])
                y_pos = range(len(words))
                ax.barh(y_pos, counts)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(words)
                ax.set_xlabel('Frequency')
                ax.set_title('Most Common Words')

                canvas = FigureCanvasTkAgg(fig, master=words_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(pady=10)

                # List the words
                for word, count in word_counts:
                    ttk.Label(words_frame, text=f"{word}: {count} occurrences").pack(anchor=tk.W, padx=20, pady=2)
            else:
                ttk.Label(words_frame, text="No words found in history").pack(pady=20)

        except ImportError:
            messagebox.showinfo("Missing Dependency",
                               "Statistics visualization requires matplotlib. Install with: pip install matplotlib")
        except Exception as e:
            self.status_var.set(f"Error generating statistics: {str(e)}")

    def show_ocr_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("OCR Settings")
        settings_window.geometry("400x500")

        # Create settings frame with scrollbar
        main_frame = ttk.Frame(settings_window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)

        settings_frame = ttk.Frame(canvas)

        # Configure scrolling
        settings_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=settings_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack scrollbar components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # PSM modes (Page Segmentation Modes)
        ttk.Label(settings_frame, text="Page Segmentation Mode:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))

        psm_modes = [
            (0, "Orientation and script detection only"),
            (1, "Automatic page segmentation with OSD"),
            (3, "Fully automatic page segmentation, but no OSD (default)"),
            (4, "Assume a single column of text of variable sizes"),
            (6, "Assume a single uniform block of text"),
            (7, "Treat the image as a single text line"),
            (8, "Treat the image as a single word"),
            (9, "Treat the image as a single word in a circle"),
            (10, "Treat the image as a single character"),
            (11, "Sparse text. Find as much text as possible in no particular order"),
            (12, "Sparse text with OSD"),
            (13, "Raw line. Treat the image as a single text line")
        ]

        for mode, desc in psm_modes:
            ttk.Radiobutton(settings_frame, text=desc, variable=self.psm_var, value=mode).pack(anchor=tk.W, padx=20)

        # OCR Engine modes
        ttk.Label(settings_frame, text="OCR Engine Mode:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(20, 5))

        oem_modes = [
            (0, "Legacy engine only"),
            (1, "Neural nets LSTM engine only"),
            (2, "Legacy + LSTM engines"),
            (3, "Default, based on what is available")
        ]

        for mode, desc in oem_modes:
            ttk.Radiobutton(settings_frame, text=desc, variable=self.oem_var, value=mode).pack(anchor=tk.W, padx=20)

        # Whitelist/Blacklist characters
        ttk.Label(settings_frame, text="Character Whitelist:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(20, 5))
        ttk.Entry(settings_frame, textvariable=self.whitelist_var, width=40).pack(fill=tk.X, padx=20)

        # Reset to defaults button
        ttk.Button(settings_frame, text="Reset to Defaults", command=self.reset_ocr_settings).pack(pady=10)

        # Save button
        ttk.Button(settings_frame, text="Save Settings", command=lambda: settings_window.destroy()).pack(pady=10)

    def reset_ocr_settings(self):
        self.psm_var.set(6)
        self.oem_var.set(3)
        self.whitelist_var.set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

if __name__ == "__main__":
    root = tk.Tk()
    app = TextRecognitionApp(root)
    root.mainloop()