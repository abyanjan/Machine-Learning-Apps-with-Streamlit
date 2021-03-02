import streamlit as st
from multiapp import MultiApp
from apps import home, data, modeling


app = MultiApp()

app.add_app('Home', home.app)
app.add_app('Data', data.app)
app.add_app('Modeling', modeling.app)

app.run()