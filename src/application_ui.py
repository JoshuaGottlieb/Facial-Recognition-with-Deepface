from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
import cv2

# Build the Kivy app
class DeepFaceApp(App):
    def build(self):
        self.layout = FloatLayout()
        
        # Camera view
        self.image = Image(size_hint=(1, 1), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.layout.add_widget(self.image)
        
        # Label for status. Text will be updated based on user actions. It should be ontop of the camera view
        self.label = Label(text='No image loaded', size_hint=(0.2, 0.1), pos_hint={'x': 0.4, 'y': 0.9})
        self.layout.add_widget(self.label)
        
        # Capture button
        capture_button = Button(text='Capture Image', size_hint=(0.15, 0.15), 
                                pos_hint={'center_x': 0.35, 'center_y': 0.1})
        capture_button.bind(on_press=self.capture_image)
        
        # Sign up button
        signup_button = Button(text='Sign Up', size_hint=(0.15, 0.15), 
                               pos_hint={'center_x': 0.65, 'center_y': 0.1})
        signup_button.bind(on_press=self.show_signup_popup)
        
        self.layout.add_widget(capture_button)
        self.layout.add_widget(signup_button)
        
        # Start the camera automatically
        self.start_camera()
        
        return self.layout

    def start_camera(self):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def capture_image(self, instance):
        if hasattr(self, 'capture'):
            self.capture.release()
            del self.capture
            Clock.unschedule(self.update)
            self.label.text = 'Capture button pressed'
        else:
            self.start_camera()

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buf = cv2.flip(frame, 0).tobytes()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = image_texture

    def show_signup_popup(self, instance):
        # Create a popup content layout when the sign up button is pressed
        popup_layout = BoxLayout(orientation='vertical')
        
        # Text inputs for name and email, can add more if needed
        self.name_input = TextInput(hint_text='Enter name')
        self.email_input = TextInput(hint_text='Enter email')
        popup_layout.add_widget(self.name_input)
        popup_layout.add_widget(self.email_input)
        
        # Button to submit the information
        submit_button = Button(text='Submit')
        submit_button.bind(on_press=self.submit_signup)
        popup_layout.add_widget(submit_button)
        
        # Create the popup
        self.popup = Popup(title='Sign Up', content=popup_layout, size_hint=(0.8, 0.8))
        self.popup.open()

    def submit_signup(self, instance):
        name = self.name_input.text
        email = self.email_input.text
        
        if name and email:
            self.label.text = f'Signed up: {name}, {email}'
            self.popup.dismiss()
        else:
            self.label.text = 'Please enter both name and email'

# Run the app
if __name__ == '__main__':
    DeepFaceApp().run()
