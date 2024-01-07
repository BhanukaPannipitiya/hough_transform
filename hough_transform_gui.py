import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
from hough_transform import hough_line, draw_lines, segment_edges

class HoughTransformGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Hough Transform GUI")

        self.image_path = None

        self.canvas = tk.Canvas(self.master)
        self.canvas.pack()

        self.load_button = tk.Button(self.master, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.process_button = tk.Button(self.master, text="Process Image", command=self.process_image)
        self.process_button.pack()

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            image = Image.open(self.image_path)
            image.thumbnail((500, 500))
            photo = ImageTk.PhotoImage(image)

            self.canvas.config(width=photo.width(), height=photo.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo

    def process_image(self):
        if self.image_path:
            image = Image.open(self.image_path).convert("L")
            img_array = np.array(image)

            accumulator, thetas, rhos, edges = hough_line(img_array)
            img_with_lines = draw_lines(img_array, accumulator, thetas, rhos)
            segmented_edges = segment_edges(img_array)

            # Display the processed images
            self.display_image(img_with_lines, "Detected 2D Lines")
            self.display_image(segmented_edges, "Segmented Edges")

    def display_image(self, img_array, title):
        # Create a new window for each processed image
        window = tk.Toplevel(self.master)
        window.title(title)

        # Convert NumPy array to ImageTk format
        img = Image.fromarray(img_array)
        img.thumbnail((500, 500))
        photo = ImageTk.PhotoImage(img)

        # Display the image in a Canvas
        canvas = tk.Canvas(window)
        canvas.pack()
        canvas.config(width=photo.width(), height=photo.height())
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = HoughTransformGUI(root)
    root.mainloop()
