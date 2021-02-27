import streamlit as st
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

def load_data(file_type='train'):
    path = 'datasets/'
    data = []
    label = []

    with open(path + file_type + '.txt', encoding='utf8') as file:
        lines = file.read()

    samples = lines.strip().split(f"\n\n{file_type}_")

    for sample in samples:
        sample = sample.strip()
        d = sample[8:-3].strip()
        t = sample[-1:]
        data.append(d)
        label.append(t)

    return data, label

data, label = load_data('train')
label = np.array(list(map(int, label)))

evaluation = list(map(lambda x: 'Bad' if x else 'Good', label))
df = pd.DataFrame({'Reviews':data, 'Evaluation':evaluation})
ProfileReport(df)

pr = ProfileReport(df, explorative=True)

st.title("Dataset Review")
st.write(df)
st_profile_report(pr)