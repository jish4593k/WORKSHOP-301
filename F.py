import tornado.ioloop
import tornado.web
import tornado.wsgi
import io
import random
import os

from tornado.log import enable_pretty_logging
from PIL import Image
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.preprocessing import image

N = 20

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        img_id = int(self.get_argument("img"))
        fragment_number = int(self.get_argument("fragment_number", random.randint(0, N - 1)))

        fn = os.path.join(os.path.dirname(__file__), "images", f"{img_id}.jpg")
        im = Image.open(fn)
        dim = im.size
        fragment_width = int(dim[0] / N)

        start_x = fragment_number * fragment_width
        end_x = (fragment_number + 1) * fragment_width

        c = im.crop((start_x, 0, end_x, dim[1]))
        c = c.convert("RGBA")

        # Image processing with PyTorch
        c_array = np.array(c)
        torch_img = torch.from_numpy(c_array)
        # Perform image processing tasks using PyTorch

        # Neural network inference with TensorFlow (Keras)
        keras_model = tf.keras.models.load_model('your_keras_model.h5')
        c_resized = c.resize((224, 224))  # Adjust size based on your model input size
        img_array = image.img_to_array(c_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        prediction = keras_model.predict(img_array)

        # Display or use the processed image and prediction results

        # Convert the processed image back to bytes for sending in response
        bio = io.BytesIO()
        c.save(bio, 'PNG')

        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Content-Type', 'image/png')
        self.set_header('X-ECE459-Fragment', str(fragment_number))
        self.write(bio.getvalue())

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    return file_path

def main():
    # You can uncomment and use this function to open a file dialog for image selection
    # selected_image_path = open_file_dialog()
    # print("Selected Image Path:", selected_image_path)

    application = tornado.web.Application([
        (r"/image", MainHandler),
    ])

    application.listen(4590)
    enable_pretty_logging()
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()
