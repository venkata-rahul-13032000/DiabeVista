import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import style
import plotly.express as px


def load_data(n):
    df = pd.read_csv('diabetes_data_upload.csv')
    if n=='1':
        st.header(f":orange[**Data Set**]")
        st.dataframe(df)
    return df


def check_mis_data(df,n):
    if n=="1":
        st.header(f":orange[**Missing Data Analysis**]")
        st.markdown(
            f'''<p class="small-font" style="font-size: 20px; color:lightgreen;font-weight:450;font-style: italic;">
                                                    <strong>This dashboard provides an overview of missing data by columns in the dataset.</strong></p>
                                                    ''',
            unsafe_allow_html=True)
        selected_columns = st.multiselect(f':orange[**_Select columns to analyze Missomg Data:_**]', df.columns)
        # Filter the dataframe based on selected columns
        if selected_columns:
            data_to_show = df[selected_columns]
        else:
            data_to_show = df
        nullity = data_to_show.isnull().astype(int)
        fig = px.imshow(nullity, text_auto=True, color_continuous_scale='YlGnBu',
                        labels=dict(color="Missing Values"))
        # title="Interactive Heatmap of Missing Values")
        st.plotly_chart(fig, use_container_width=True)

# @st.cache_resource(ttl=43200, max_entries=1000)
def corrleation_data(dfa,n):
    dfa['Gender'] = dfa['Gender'].map({'Male': 1, 'Female': 0})
    dfa['class'] = dfa['class'].map({'Positive': 1, 'Negative': 0})
    dfa['Polyuria'] = dfa['Polyuria'].map({'Yes': 1, 'No': 0})
    dfa['Polydipsia'] = dfa['Polydipsia'].map({'Yes': 1, 'No': 0})
    dfa['sudden weight loss'] = dfa['sudden weight loss'].map({'Yes': 1, 'No': 0})
    dfa['weakness'] = dfa['weakness'].map({'Yes': 1, 'No': 0})
    dfa['Polyphagia'] = dfa['Polyphagia'].map({'Yes': 1, 'No': 0})
    dfa['Genital thrush'] = dfa['Genital thrush'].map({'Yes': 1, 'No': 0})
    dfa['visual blurring'] = dfa['visual blurring'].map({'Yes': 1, 'No': 0})
    dfa['Itching'] = dfa['Itching'].map({'Yes': 1, 'No': 0})
    dfa['Irritability'] = dfa['Irritability'].map({'Yes': 1, 'No': 0})
    dfa['delayed healing'] = dfa['delayed healing'].map({'Yes': 1, 'No': 0})
    dfa['partial paresis'] = dfa['partial paresis'].map({'Yes': 1, 'No': 0})
    dfa['muscle stiffness'] = dfa['muscle stiffness'].map({'Yes': 1, 'No': 0})
    dfa['Alopecia'] = dfa['Alopecia'].map({'Yes': 1, 'No': 0})
    dfa['Obesity'] = dfa['Obesity'].map({'Yes': 1, 'No': 0})
    if n=="1":
        corrdata = dfa.corr()
        st.header(f":orange[**Exploratory Data Analysis**]")
        ax, fig = plt.subplots(figsize=(15, 8))
        sns.heatmap(corrdata, annot=True, cmap="YlGnBu")
        st.pyplot(ax)
    return dfa

if __name__ == "__main__":
    style.Style(title='Data Analysis', header=f"")
    df = load_data('1')
    check_mis_data(df,'1')
    df_cor = corrleation_data(df, '1')
    # st.link_button("Home", "http://localhost:8501/Feature%20Extraction")
#Mapping or Converting the text into values

