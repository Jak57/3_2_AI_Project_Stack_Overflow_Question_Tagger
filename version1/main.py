import streamlit as st
from streamlit_quill import st_quill # text editor
from streamlit_ace import st_ace # for showing code editor
from streamlit_ace import st_ace, KEYBINDINGS, LANGUAGES, THEMES

import numpy as np
from sklearn import datasets
from PIL import Image # for displaying image

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC # support vector classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd # importing pandas
#import pickle # for saving/loading ML model

# from ai_project import final_prediction # commenting here

st.title(" :pencil: Stack Overflow Question Tagger")

st.write("""
## :question: Ask a public question
""")

def sidebar_content():
    # showing stack overflow logo
    image = Image.open('D:\streamlit2\images\stack_overflow_logo.png')
    st.sidebar.image(image)

    dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast cancer", "Wine Dataset"))
    classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))
    return dataset_name, classifier_name

dataset_name, classifier_name = sidebar_content()

# content in the main page
# showing rules for asking questions
st.info('''
:mag_right: **Writing a good question**

You are ready to ask a programming-related question and this form will help guide
you through the process.

:page_with_curl: Steps
+ Summarize your problem in a one-line title.
+ Describe your problem in more detail.
+ Describe what you tried and what you expected to happen.
+ Review your question and post it to the site.

''')

st.subheader('''
Title
''')

question = ""
q1 = ""
q2 = ""
q3 = ""

topic = st.text_input(
    label="Be specific and imagine you’re asking a question to another person.",
    placeholder="e.g. Is there any R function for finding the index of an element in a vector?")

title = str(topic)

#question += (" " + str(topic))
#st.write(q1)

st.subheader('''
What are the details of your problem?
''')
st.write("Introduce the problem and expand on what you put in the title. Minimum 20 characters.")

def problem_description():
    """
    Returns questions description
    """
    st.write("Write your text here:")
    content = st_quill(
        placeholder="Write your text here",
        key="quill"
    )
    st.markdown("---")

    if content:
        #st.write(content)
        #q2 = str(content)
        return str(content)
        # global question = question + (" " + str(content))
    else:
        return ""

description = " " + problem_description()

def code_editor():
    """
    Showing code and text editor
    """
    # st.write("Write your text here:")
    # content = st_quill(
    #     placeholder="Write your text here",
    #     key="quill"
    # )
    # st.markdown("---")

    # if content:
    #     st.write(content)
    #     q2 = str(content)
    #     global question = question + (" " + str(content))

    c1, c2 = st.columns([3, 1])
    c2.subheader("Parameters")

    with c1:
        st.write("Write your code here:")
        content5 = st_ace(
            placeholder=c2.text_input("Editor placeholder", value="Write your code here"),
            #language=c2.selectbox("Language mode", options=LANGUAGES, index=121),
            theme=c2.selectbox("Theme", options=THEMES, index=30),
            keybinding=c2.selectbox("Keybinding mode", options=KEYBINDINGS, index=3),
            font_size=c2.slider("Font size", 5, 24, 14),
            show_gutter=c2.checkbox("Show gutter", value=True),
            show_print_margin=c2.checkbox("Show print margin", value=False),
            min_lines=35,
            key="ace",
        )

        if content5:
            st.subheader("Content")
            st.text(content5)
            q3 = str(content5)
            #question += (" " + content5)
            return str(content5)
        else:
            return ""

problem_code = " " + code_editor()

#question = q1 + q2 + q3

#st.write(question)

def text_and_code_editor2():
    """
    Showing code and text editor
    """
    st.write("Write your text here:")
    content6 = st_quill(
        placeholder="Write your text here",
        key="quill6"
    )
    st.markdown("---")

    if content6:
        st.write(content6)

    c1, c2 = st.columns([3, 1])
    c2.subheader("Parameters")

    with c1:
        st.write("Write your code here:")
        content7 = st_ace(
            placeholder=c2.text_input("Editor placeholder", value="Write your code here "),
            language=c2.selectbox("Language mode ", options=LANGUAGES, index=121),
            theme=c2.selectbox("Theme ", options=THEMES, index=30),
            keybinding=c2.selectbox("Keybinding mode ", options=KEYBINDINGS, index=3),
            font_size=c2.slider("Font size ", 5, 24, 14),
            show_gutter=c2.checkbox("Show gutter ", value=True),
            show_print_margin=c2.checkbox("Show print margin ", value=False),
            min_lines=35,
            key="ace ",
        )

        if content7:
            st.subheader("Content")
            st.text(content7)

st.subheader('''
What did you try and what were you expecting?
''')
st.write("Describe what you tried, what you expected to happen, and what actually resulted. Minimum 20 characters.")

with st.expander("Expand"):
    text_and_code_editor2()

st.button("Predict Tag", type="primary")

q="My summation code is not working. How to do summation in stl?"

#pred = final_prediction(q)
#st.write(pred[0])


st.write(question)

st.write("q1 ", q1)
st.write("q2 ", q2)
st.write("q3 ", q3)

st.write(problem_code)

question = title + description + problem_code

st.write("full ", question)