import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn import preprocessing , metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
import style


# @st.cache_resource(ttl=43200, max_entries=1000)
def load_pt_data():
    # read the file and create a pandas dataframe
    data = pd.read_csv('diabetic_data.csv')

    st.write(data.head(10))

    return data
def admis_deatils(data):
    readmitted_counts = data['readmitted'].value_counts()
    total_count = readmitted_counts.sum()
    percentages = (readmitted_counts / total_count) * 100
    fig = go.Figure(data=[
        go.Bar(x=readmitted_counts.index, y=readmitted_counts.values)
    ])
    for i, value in enumerate(percentages):
        fig.add_annotation(
            x=readmitted_counts.index[i],
            y=readmitted_counts.values[i],
            text=f'{value:.2f}%',
            showarrow=True,
            font=dict(size=12),
            yshift=10
        )
    fig.update_layout(
        title='Readmission details',
        xaxis_title='Readmission within 30 days and after 30 days',
        yaxis_title='Count'
    )
    st.plotly_chart(fig, use_container_width=True)

def data_preprocess(data):
    data['readmitted'] = pd.Series([0 if val == 'NO' else 1 for val in data['readmitted']])
    data_origin = data
    # remove irrelevant features
    data.drop(['encounter_id', 'patient_nbr', 'payer_code'], axis=1, inplace=True)
    # check NA in 'weight'
    # data[data['weight'] == '?'].shape[0] * 1.0 / data.shape[0]
    # # check NA in 'medical_specialty'
    # data[data['medical_specialty'] == '?'].shape[0] * 1.0 / data.shape[0]
    # remove 'weight' and 'medical_specialty' because it's hard to do imputation on them beacuse "?" missing values
    data.drop(['weight', 'medical_specialty'], axis=1, inplace=True)
    # remove rows that have NA in 'race', 'diag_1', 'diag_2', or 'diag_3'
    # remove rows that have invalid values in 'gender'
    data = data[data['race'] != '?']
    data = data[data['diag_1'] != '?']
    data = data[data['diag_2'] != '?']
    data = data[data['diag_3'] != '?']
    data = data[data['gender'] != 'Unknown/Invalid']
    data.groupby('age').size().plot(kind='bar')
    age_counts = data['age'].value_counts()
    total_count = age_counts.sum()
    percentages = (age_counts / total_count) * 100
    fig = go.Figure(data=[
        go.Bar(x=age_counts.index, y=age_counts.values)
    ])
    for i, value in enumerate(percentages):
        fig.add_annotation(
            x=age_counts.index[i],
            y=age_counts.values[i],
            text=f'{value:.2f}%',
            showarrow=False,
            font=dict(size=12),
            yshift=10
        )
    fig.update_layout(
        title='Age distribution',
        xaxis_title='Age',
        yaxis_title='Count'
    )
    st.plotly_chart(fig, use_container_width=True)

    # plt.ylabel('Count')
    # Recategorize 'age' so that the population is more evenly distributed
    data['age'] = pd.Series(['[0-50)' if val in ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)'] else val
                             for val in data['age']], index=data.index)
    data['age'] = pd.Series(['[80-100)' if val in ['[80-90)', '[90-100)'] else val
                             for val in data['age']], index=data.index)

    # data.groupby('age').size().plot(kind='bar')
    # plt.ylabel('Count')
    return data
def medications(data):
    # original 'discharge_disposition_id' contains 28 levels
    # reduce 'discharge_disposition_id' levels into 2 categories
    # discharge_disposition_id = 1 corresponds to 'Discharge Home'
    data['discharge_disposition_id'] = pd.Series(['Home' if val == 1 else 'Other discharge'
                                                  for val in data['discharge_disposition_id']], index=data.index)
    # original 'admission_source_id' contains 25 levels
    # reduce 'admission_source_id' into 3 categories
    data['admission_source_id'] = pd.Series(
        ['Emergency Room' if val == 7 else 'Referral' if val == 1 else 'Other source'
         for val in data['admission_source_id']], index=data.index)
    # original 'admission_type_id' contains 8 levels
    # reduce 'admission_type_id' into 2 categories
    data['admission_type_id'] = pd.Series(['Emergency' if val == 1 else 'Other type'
                                           for val in data['admission_type_id']], index=data.index)

    fig = make_subplots(rows=2, cols=2, subplot_titles=('miglitol', 'nateglinide', 'acarbose', 'insulin'))

    # Group and plot 'miglitol'
    fig.add_trace(
        go.Bar(x=data['miglitol'].value_counts().index, y=data['miglitol'].value_counts().values, name='miglitol'),
        row=1, col=1)
    # Group and plot 'nateglinide'
    fig.add_trace(go.Bar(x=data['nateglinide'].value_counts().index, y=data['nateglinide'].value_counts().values,
                         name='nateglinide'), row=1, col=2)
    # Group and plot 'acarbose'
    fig.add_trace(
        go.Bar(x=data['acarbose'].value_counts().index, y=data['acarbose'].value_counts().values, name='acarbose'),
        row=2, col=1)
    # Group and plot 'insulin'
    fig.add_trace(
        go.Bar(x=data['insulin'].value_counts().index, y=data['insulin'].value_counts().values, name='insulin'), row=2,
        col=2)

    # Update layout for better visualization
    fig.update_layout(
        title='Comparison of Diabetes Medications',

    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
    # keep only 'insulin' and remove the other 22 diabetes medications
    data.drop(['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
               'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
               'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
               'citoglipton', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
               'metformin-rosiglitazone', 'metformin-pioglitazone'], axis=1, inplace=True)
    data['diag_1'] = pd.Series([1 if val.startswith('250') else 0 for val in data['diag_1']], index=data.index)
    data.drop(['diag_2', 'diag_3'], axis=1, inplace=True)
    return data
@st.cache_resource(ttl=43200, max_entries=1000)
def model_slection(X_cv, X_test, y_cv, y_test):
    clf1 = RandomForestClassifier()
    RF_score = cross_val_score(clf1, X_cv, y_cv, cv=10, scoring='accuracy').mean()

    clf2 = GaussianNB()
    NB_score = cross_val_score(clf2, X_cv, y_cv, cv=10, scoring='accuracy').mean()

    clf3 = LogisticRegression()
    LR_score = cross_val_score(clf3, X_cv, y_cv, cv=10, scoring='accuracy').mean()
    # Logistic Regression on Top 6 features
    # still be able to achieve good result with reduced running time
    LR_score_top = cross_val_score(clf3, X_cv_top6, y_cv, cv=10, scoring='accuracy').mean()
    x_axis = np.arange(4)
    y_axis = [RF_score, NB_score, LR_score,LR_score_top]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Random Forest', 'Naive Bayes', 'Logistic Regression','LR on Top 6 Features'], y=y_axis, width=0.2))
    fig.update_layout(
        title='Comparison of Cross-Validated Accuracy among Models',
        xaxis_title='Model',
        yaxis_title='Cross-Validated Accuracy',
        width=800,
        height=600
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
@st.cache_resource(ttl=43200, max_entries=1000)
def data_prep_feture_sle(data):
    # one-hot-encoding on categorical features
    # convert nominal values to dummy values
    df_age = pd.get_dummies(data['age'])
    df_race = pd.get_dummies(data['race'])
    df_gender = pd.get_dummies(data['gender'])
    df_max_glu_serum = pd.get_dummies(data['max_glu_serum'])
    df_A1Cresult = pd.get_dummies(data['A1Cresult'])
    df_insulin = pd.get_dummies(data['insulin'])
    df_change = pd.get_dummies(data['change'])
    df_diabetesMed = pd.get_dummies(data['diabetesMed'])
    df_discharge_disposition_id = pd.get_dummies(data['discharge_disposition_id'])
    df_admission_source_id = pd.get_dummies(data['admission_source_id'])
    df_admission_type_id = pd.get_dummies(data['admission_type_id'])

    data = pd.concat([data, df_age, df_race, df_gender, df_max_glu_serum, df_A1Cresult,
                      df_insulin, df_change, df_diabetesMed, df_discharge_disposition_id,
                      df_admission_source_id, df_admission_type_id], axis=1)
    data.drop(['age', 'race', 'gender', 'max_glu_serum', 'A1Cresult', 'insulin', 'change',
               'diabetesMed', 'discharge_disposition_id', 'admission_source_id',
               'admission_type_id'], axis=1, inplace=True)
    # apply square root transformation on right skewed count data to reduce the effects of extreme values.
    # here log transformation is not appropriate because the data is Poisson distributed and contains many zero values.
    data['number_outpatient'] = data['number_outpatient'].apply(lambda x: np.sqrt(x + 0.5))
    data['number_emergency'] = data['number_emergency'].apply(lambda x: np.sqrt(x + 0.5))
    data['number_inpatient'] = data['number_inpatient'].apply(lambda x: np.sqrt(x + 0.5))
    # feature scaling, features are standardized to have zero mean and unit variance
    feature_scale_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                          'number_diagnoses', 'number_inpatient', 'number_emergency', 'number_outpatient']
    scaler = preprocessing.StandardScaler().fit(data[feature_scale_cols])
    data_scaler = scaler.transform(data[feature_scale_cols])

    data_scaler_df = pd.DataFrame(data=data_scaler, columns=feature_scale_cols, index=data.index)
    data.drop(feature_scale_cols, axis=1, inplace=True)
    data = pd.concat([data, data_scaler_df], axis=1)
    # create X (features) and y (response)
    X = data.drop(['readmitted'], axis=1)
    y = data['readmitted']
    # st.write("dibov",X)
    # st.write(y)
    # split X and y into cross-validation (75%) and testing (25%) data sets
    from sklearn.model_selection import train_test_split
    X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.25)
    # st.write((len(X_cv)),"afisng")
    # fit Random Forest model to the cross-validation data
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier()
    forest.fit(X_cv, y_cv)
    importances = forest.feature_importances_

    # make importance relative to the max importance
    feature_importance = 100.0 * (importances / importances.max())
    sorted_idx = np.argsort(feature_importance)
    feature_names = list(X_cv.columns.values)
    feature_names_sort = [feature_names[indice] for indice in sorted_idx]
    pos = np.arange(sorted_idx.shape[0]) + .5
    # st.write('Top 6 features are: ')
    # for feature in feature_names_sort[::-1][:6]:
    #     st.write(feature)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=feature_importance[sorted_idx], y=feature_names_sort, orientation='h'))

    # Update layout for better visualization
    fig.update_layout(
        title='Relative Feature Importance',
        xaxis_title='Importance (%)',
        yaxis_title='Feature',
        width=600,
        height=600
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
    # make a smaller feature set which only contains the top 6 features
    X_cv_top6 = X_cv[['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses',
                      'number_inpatient']]
    # st.write(len(X_cv), len(X_test), len(y_cv), len(y_test), len(X_cv_top6))
    return X_cv, X_test, y_cv, y_test, X_cv_top6 ,X,y
def print_confusion_matrix():
    st.markdown(
        f'''<p class="small-font" style="font-size: 18px;font-weight:450;text-align:justify;">
            <strong style="color:orange;font-size:22px;">Confusion Matrix:</strong>
            <ul style ="font-size:18px;" >
                It describes the performance of a classification model.<br>
                &emsp;&emsp;True Positives (TP): we correctly predicted that they do have diabetes.<br>
                &emsp;&emsp;True Negatives (TN): we correctly predicted that they don't have diabetes.<br>
                &emsp;&emsp;False Positives (FP): we incorrectly predicted that they do have diabetes (a "Type I error").<br>
                &emsp;&emsp;False Negatives (FN): we incorrectly predicted that they don't have diabetes (a "Type II error").<br>
            </ul>
            </p>
            ''',
        unsafe_allow_html=True
    )
@st.cache_resource(ttl=43200, max_entries=1000)
def parameter_tuning_grid(X ,y):
    # define the parameter values that should be searched
    C_range = np.arange(0.1, 3.1, 0.2)
    param_grid = dict(C=C_range)
    clf = LogisticRegression()
    # Split data into training and cross-validation sets
    X_cv, X_test, y_cv, y_test = train_test_split(X, y, test_size=0.25)
    # Feature scaling
    scaler = StandardScaler()
    X_cv_scaled = scaler.fit_transform(X_cv)
    # Grid search with cross-validation
    grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
    grid.fit(X_cv_scaled, y_cv)

    logreg = LogisticRegression(C=grid.best_params_['C'])
    logreg.fit(X_cv, y_cv)

    y_pred_class = logreg.predict(X_test)

    confusion = metrics.confusion_matrix(y_test, y_pred_class)
    return confusion ,logreg

if __name__ == "__main__":
    style.Style(title='Home', header=f"")
    df = load_pt_data()
    # data.check_mis_data(df,"1")
    col1,col2=st.columns([0.8,1.2])
    with col1:
        admis_deatils(df)
    with col2:
        df_pre=data_preprocess(df)
    df_pre=medications(df_pre)
    data_res =data_prep_feture_sle(df_pre)
    # st.write(len(X_cv), len(X_test), len(y_cv), len(y_test), len(X_cv_top6))
    X_cv=data_res[0]
    X_test=data_res[1]
    y_cv=data_res[2]
    y_test=data_res[3]
    X_cv_top6=data_res[4]
    X=data_res[5]
    y=data_res[6]
    model_slection(X_cv, X_test, y_cv, y_test)
    confusion,logreg=parameter_tuning_grid(X , y)
    print_confusion_matrix()
    # st.write(confusion)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    # st.write(TP,TN,FP,FN)
    bar_trace = go.Bar(
        x=["True Positives", "True Negatives", "False Positives", "False Negatives"],
        y=[TP, TN, FP, FN],
        marker_color=["green", "blue", "red", "orange"],
    )
    layout = go.Layout(title="Confusion Matrix Metrics")
    fig = go.Figure(data=[bar_trace], layout=layout)
    st.plotly_chart(fig,use_container_width=True)
    # Store the predicted probabilities for class 1
    y_pred_prob = logreg.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    # Create ROC curve trace
    roc_trace = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve')
    # Create layout
    layout = go.Layout(
            title='ROC curve for diabetes readmission',
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity)',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    # Create figure
    fig = go.Figure(data=[roc_trace], layout=layout)
    # Add explanation text
    explanation = (
            "Receiver operating characteristic (ROC) curve and area under the curve (AUC)."
            "ROC is a graphical plot which illustrates the performance of a binary classifier system as its discrimination threshold is varied. "
            "AUC is the percentage of the ROC plot that is underneath the curve. AUC is useful as a single number summary of classifier performance."
        )
    st.write(f":orange[**_{explanation}_**]")
    st.plotly_chart(fig, use_container_width=True)

