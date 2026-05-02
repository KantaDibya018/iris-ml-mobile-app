import os
import numpy as np
import joblib

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button


class IrisMobileApp(App):

    def build(self):
        # --- Load model ONCE (important for performance) ---
        model_path = os.path.join(os.getcwd(), 'model_rf.pkl')
        self.model = joblib.load(model_path)

        # --- UI Layout ---
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=10)

        # Title
        self.layout.add_widget(Label(text="Iris Flower Predictor", font_size=28))

        # Inputs
        self.sl = TextInput(hint_text="Sepal Length", multiline=False, input_filter='float')
        self.sw = TextInput(hint_text="Sepal Width", multiline=False, input_filter='float')
        self.pl = TextInput(hint_text="Petal Length", multiline=False, input_filter='float')
        self.pw = TextInput(hint_text="Petal Width", multiline=False, input_filter='float')

        self.layout.add_widget(self.sl)
        self.layout.add_widget(self.sw)
        self.layout.add_widget(self.pl)
        self.layout.add_widget(self.pw)

        # Button
        btn = Button(text="Predict Now", size_hint=(1, 0.3))
        btn.bind(on_press=self.do_prediction)
        self.layout.add_widget(btn)

        # Result
        self.result_lbl = Label(text="Result: Waiting...", font_size=20)
        self.layout.add_widget(self.result_lbl)

        return self.layout

    def do_prediction(self, instance):
        try:
            # --- Input validation ---
            sl = float(self.sl.text)
            sw = float(self.sw.text)
            pl = float(self.pl.text)
            pw = float(self.pw.text)

            data = np.array([[sl, sw, pl, pw]])

            # --- Prediction ---
            res = self.model.predict(data)[0]

            names = ["Setosa", "Versicolor", "Virginica"]
            self.result_lbl.text = f"Result: {names[res]}"

        except ValueError:
            self.result_lbl.text = "⚠️ Please enter all values correctly"


if __name__ == '__main__':
    IrisMobileApp().run()