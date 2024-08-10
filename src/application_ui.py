from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.graphics import Color, RoundedRectangle
import cv2
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from modules.face_api import init_model, find_image_match
import os
import re

class RoundedButton(ButtonBehavior, Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0, 0, 0, 0)
        self.background_normal = ''
        self.bind(size=self.update_canvas, pos=self.update_canvas)

    def update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0.2, 0.6, 1, 1)  # Set button color (light blue)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[20, ])

    def on_press(self):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0.1, 0.5, 0.9, 1)  # Slightly darker when pressed
            RoundedRectangle(pos=self.pos, size=self.size, radius=[20, ])

    def on_release(self):
        self.update_canvas()

class DeepFaceApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize Firebase
        cred = credentials.Certificate("./serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.model = init_model('../pretrained_models/VGGFace2_DeepFace_weights_val-0.9034.h5')
        # Get all documents from the 'vectors' collection
        snapshots = self.db.collection('vectors').get()
        docs = [x.to_dict() for x in snapshots]
        self.vector_dict = {list(x.keys())[0]:list(x.values())[0] for x in docs}
        self.match_result = None

    def build(self):
        self.layout = FloatLayout()

        # Camera view
        self.image = Image(size_hint=(1, 1), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.layout.add_widget(self.image)

        # Label for status
        self.label = Label(text='No image loaded', size_hint=(0.8, 0.1), pos_hint={'center_x': 0.5, 'top': 1})
        self.layout.add_widget(self.label)

        # Match result label
        self.match_label = Label(text='', size_hint=(0.8, 0.1), pos_hint={'center_x': 0.5, 'y': 0.05})
        # self.layout.add_widget(self.match_label)

        # Matched image view
        self.matched_image = Image(size_hint=(0.3, 0.3), pos_hint={'center_x': 0.5, 'y': 0.35})

        # Capture button
        capture_button = RoundedButton(text='Capture Image', size_hint=(0.2, 0.1), pos_hint={'center_x': 0.5, 'y': 0.02})
        capture_button.bind(on_press=self.capture_image)

        ## Sign up button
        #signup_button = RoundedButton(text='Sign Up', size_hint=(0.2, 0.1), pos_hint={'center_x': 0.7, 'y': 0.02})
        #signup_button.bind(on_press=self.show_signup_popup)

        self.layout.add_widget(capture_button)
        #self.layout.add_widget(signup_button)

        # Start the camera automatically
        self.start_camera()
        return self.layout

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            #SIMPLEX, 1, (0, 255, 0), 2
            # Convert it to texture
            buf = cv2.flip(frame, 0).tobytes()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # Display image from the texture
            self.image.texture = image_texture

    def capture_image(self, instance):
        if hasattr(self, 'capture'):
            ret, frame = self.capture.read()
            if ret:
                # Save the captured image
                cv2.imwrite("captured_image.jpg", frame)
                self.process_captured_image()
            else:
                self.label.text = 'Failed to capture image'
        else:
            self.start_camera()

    def process_captured_image(self):
        # Match with vectors in Firestore
        self.match_vector()

        if type(self.match_result) != str:
            self.label.text = f"Match found: {self.match_result[0]}"
            self.match_label.text = f"Distance: {self.match_result[1]}"
            self.display_matched_image(self.match_result[0])
        else:
            self.label.text = "No match found"
            self.match_label.text = f"{self.match_result}"
            self.clear_matched_image()

    def clear_matched_image(self):
        # Clear the texture of the matched image
        self.matched_image.texture = None

    def match_vector(self):       
        results = find_image_match("captured_img.jpg", self.vector_dict,
                                   model = self.model, shape_predictor_path = './pretrained_models/shape_predictor_5_face_landmarks.dat')
        
        if type(results) == str:
            self.match_result = results
            return
        
        self.match_result = results[0][0]
        return

    def extract_image_filename(text):
        # Split the text into parts
        parts = text.split()
        # Iterate over each part to find the one ending with an image file extension
        for part in parts:
            if part.endswith('.jpg') or part.endswith('.png') or part.endswith('.jpeg'):
                return part
        # Return None if no image file name is found
        return None

    def extract_image_filename(self, text):
        # Regular expression to match the image file name ending with .jpg
        match = re.search(r'([\w\-]+\.jpg)', text)
        if match:
            return match.group(1)
        return None

    def display_matched_image(self, image_name):
        print('image_name', image_name)
        image_file_name = self.extract_image_filename(image_name)
        image_path = os.path.join('../data/dummy_dataset/input_images', image_file_name)
        if os.path.exists(image_path):
            matched_frame = cv2.imread(image_path)
            buf = cv2.flip(matched_frame, 0).tobytes()
            image_texture = Texture.create(size=(matched_frame.shape[1], matched_frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.matched_image.texture = image_texture
            self.matched_image.pos_hint = {'right': 1, 'top': 1}

            # Check if the widget already has a parent
            if self.matched_image.parent:
                self.matched_image.parent.remove_widget(self.matched_image)

            self.layout.add_widget(self.matched_image)


    #def show_signup_popup(self, instance):
    #    popup_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
    #    self.name_input = TextInput(hint_text='Enter name')
    #    self.email_input = TextInput(hint_text='Enter email')
    #    popup_layout.add_widget(self.name_input)
    #    popup_layout.add_widget(self.email_input)
    #    submit_button = RoundedButton(text='Submit', size_hint=(1, 0.3))
    #    submit_button.bind(on_press=self.submit_signup)
    #    popup_layout.add_widget(submit_button)
    #    self.popup = Popup(title='Sign Up', content=popup_layout, size_hint=(0.8, 0.4))
    #    self.popup.open()

    #def submit_signup(self, instance):
    #    name = self.name_input.text
    #    email = self.email_input.text
    #    if name and email:
    #        self.label.text = f'Signed up: {name}, {email}'
    #        self.popup.dismiss()
    #    else:
    #        self.label.text = 'Please enter both name and email'

if __name__ == '__main__':
    DeepFaceApp().run()