import pandas as pd
from sklearn.feature_selection import SelectKBest,VarianceThreshold,chi2
import streamlit as st
import style
import plotly.express as px
import data


def graph(df_corr):
    # st.write(df_corr)
    col1,col2,col3=st.columns(3)
    with col1:
        fig = px.histogram(df_corr, x='Age', nbins=30)
        st.plotly_chart(fig, use_container_width=True)

        #polyura
        fig = px.scatter(df_corr, x="Polyuria", y="class", title='Point Plot of Polyuria vs Class',
                         trendline='ols')
        st.plotly_chart(fig ,use_container_width=True)
        #Alopecia
        # st.write("")
        fig = px.bar(df_corr, x='class', y='Alopecia', title='Bar Plot of Alopecia by Class')
        st.plotly_chart(fig, use_container_width=True)

        #Obesity
        # st.write("")

    with col2:
        # fig, ax = plt.subplots(figsize=(10, 6))
        # sns.barplot(x='class', y='Age', data=df_corr, ax=ax)
        # ax.set_title('Average Age per Class')
        # ax.set_xlabel('Class')
        # ax.set_ylabel('Average Age')
        # st.pyplot(fig)
        # ds = df_corr['class'].value_counts().reset_index()
        #
        # # Display class distribution
        # st.write("Class Distribution:")
        # st.write(ds)
        # st.write("1 (Positive) & 0 (Negative)")

        #Polydipsia
        fig = px.bar(df_corr, x='Polydipsia', y='class', title='Bar Plot of Polydipsia vs Class')
        st.plotly_chart(fig,use_container_width=True)

        #visual blurring
        # st.write("")
        fig = px.bar(df_corr, x='class', y='visual blurring', title='Bar Plot of visual blurring by Class')
        st.plotly_chart(fig, use_container_width=True)

        #Irritability
        # st.write("")
        fig = px.bar(df_corr, x='class', y='Irritability', title='Bar Plot of Irritability by Class')
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        # fig, ax = plt.subplots(figsize=(10, 13))
        # sns.countplot(x='class', data=df_corr, hue='Gender')
        # st.pyplot(fig,use_container_width=True)
        #partial paresis
        # st.write("")
        fig = px.bar(df_corr, x='class', y='partial paresis', title='Bar Plot of Partial Paresis by Class')
        st.plotly_chart(fig, use_container_width=True)

        #Itching
        # st.write("")
        fig = px.bar(df_corr, x='class', y='Itching', title='Bar Plot of Itching by Class')
        st.plotly_chart(fig, use_container_width=True)
        #obesity
        fig = px.bar(df_corr, x='class', y='Obesity', title='Bar Plot of Obesity by Class')
        st.plotly_chart(fig, use_container_width=True)

def feature_selection(dfa):
    X1 = dfa.iloc[:, 0:-1]
    y1 = dfa.iloc[:, -1]
    best_feature = SelectKBest(score_func=chi2, k=10)
    fit = best_feature.fit(X1, y1)
    dataset_scores = pd.DataFrame(fit.scores_)
    dataset_cols = pd.DataFrame(X1.columns)
    featurescores = pd.concat([dataset_cols, dataset_scores], axis=1)
    featurescores.columns = ['column', 'scores']
    # Sort the DataFrame by scores
    featurescores_sorted = featurescores.sort_values(by='scores', ascending=False)

    # Create a bar plot using Plotly Express
    fig = px.bar(featurescores_sorted, x='column', y='scores',
                 title='Feature Scores from SelectKBest',
                 labels={'column': 'Features', 'scores': 'Scores'})

    # Rotate x-axis labels for better readability
    fig.update_layout(xaxis_tickangle=-45)
    st.markdown(
        f'''<p class="small-font" style="font-size: 20px; color:orange;font-weight:450;">
                                                <strong>Select top 10 features based on the scores</strong></p>
                                                ''',
        unsafe_allow_html=True)
    st.plotly_chart(fig,use_container_width=True)
def variance_each_feature(dfa):
    X1 = dfa.iloc[:, 0:-1]
    y1 = dfa.iloc[:, -1]
    st.write(f":orange[**Checking the variance of each feature**]")
    feature_high_variance = VarianceThreshold(threshold=(0.5 * (1 - 0.5)))
    falls = feature_high_variance.fit(X1)
    dfa_scores1 = pd.DataFrame(falls.variances_)
    dat1 = pd.DataFrame(X1.columns)
    high_variance = pd.concat([dfa_scores1, dat1], axis=1)
    high_variance.columns = ['variance', 'cols']
    high_variance_filtered=high_variance[high_variance['variance'] > 0.2]
    # Create a bar plot using Plotly Express
    fig = px.bar(round(high_variance_filtered,2), x='cols', y='variance',
                 title='Feature Variance',
                 labels={'cols': 'Feature', 'variance': 'Variance'},text='variance')

    # Rotate x-axis labels for better readability
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig,use_container_width=True)
if __name__ == "__main__":
    style.Style(title='Data Analysis', header=f"Sample Plots")
    df = data.load_data("0")
    df_corr=data.corrleation_data(df,'0')
    graph(df_corr)
    feature_selection(df_corr)
    variance_each_feature(df_corr)
