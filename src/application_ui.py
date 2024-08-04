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
from modules.face_api import get_image_vector, init_model


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
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        self.model = init_model()
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
        self.layout.add_widget(self.match_label)

        # Capture button
        capture_button = RoundedButton(text='Capture Image', size_hint=(0.2, 0.1),
                                       pos_hint={'center_x': 0.3, 'y': 0.02})
        capture_button.bind(on_press=self.capture_image)

        # Sign up button
        signup_button = RoundedButton(text='Sign Up', size_hint=(0.2, 0.1),
                                      pos_hint={'center_x': 0.7, 'y': 0.02})
        signup_button.bind(on_press=self.show_signup_popup)

        self.layout.add_widget(capture_button)
        self.layout.add_widget(signup_button)

        # Start the camera automatically
        self.start_camera()

        return self.layout

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Add match result text to the frame if a match was found
            if self.match_result:
                cv2.putText(frame, f"Match found: {self.match_result}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
        # Get vector for captured image
        captured_vector = get_image_vector("captured_image.jpg", self.model)
        
        # Match with vectors in Firestore
        self.match_result = self.match_vector(captured_vector)
        
        if self.match_result:
            self.label.text = f"Match found: {self.match_result}"
            self.match_label.text = f"Match found: {self.match_result}"
        else:
            self.label.text = "No match found"
            self.match_label.text = "No match found"

    def match_vector(self, captured_vector):
        threshold = 0.6  # Adjust this threshold as needed
        
        # Get all documents from the 'vectors' collection
        docs = self.db.collection('vectors').get()
        
        for doc in docs:
            data = doc.to_dict()
            for key, stored_vector in data.items():
                similarity = self.cosine_similarity(captured_vector, np.array(stored_vector))
                if similarity > threshold:
                    return key.split('+')[0]  # Return the image name
        
        return None

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def show_signup_popup(self, instance):
        popup_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        self.name_input = TextInput(hint_text='Enter name')
        self.email_input = TextInput(hint_text='Enter email')
        popup_layout.add_widget(self.name_input)
        popup_layout.add_widget(self.email_input)

        submit_button = RoundedButton(text='Submit', size_hint=(1, 0.3))
        submit_button.bind(on_press=self.submit_signup)
        popup_layout.add_widget(submit_button)

        self.popup = Popup(title='Sign Up', content=popup_layout, size_hint=(0.8, 0.4))
        self.popup.open()

    def submit_signup(self, instance):
        name = self.name_input.text
        email = self.email_input.text

        if name and email:
            self.label.text = f'Signed up: {name}, {email}'
            self.popup.dismiss()
        else:
            self.label.text = 'Please enter both name and email'

if __name__ == '__main__':
    DeepFaceApp().run()