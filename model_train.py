from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from statistics import mode
from sklearn.svm import SVC
import streamlit as st
import style
import data


@st.cache_resource(ttl=43200, max_entries=1000)
def logistic_regression(X_train, X_test, y_train, y_test):
    # Train the logistic regression model
    lg = LogisticRegression()
    lg.fit(X_train, y_train)
    # Perform cross-validation
    accuracies = cross_val_score(estimator=lg, X=X_train, y=y_train, cv=10)
    # Calculate mean accuracy and standard deviation
    mean_accuracy = accuracies.mean() * 100
    std_deviation = accuracies.std() * 100
    # Make predictions on the test set
    pre = lg.predict(X_test)
    # Calculate accuracy on the test set
    logistic_regression_accuracy = accuracy_score(pre, y_test)
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(pre, y_test)
    # Create a line graph to visualize mean accuracy and standard deviation
    fig = go.Figure(data=[
        go.Bar(x=['Mean Accuracy from cross-validation', 'Standard Deviation from cross-validation ',
                  'Accuracy on Test Set'],
               y=[mean_accuracy, std_deviation, logistic_regression_accuracy * 100],
               marker=dict(color=['blue', 'green', 'red']))
    ])
    # Update layout for better visualization
    fig.update_layout(title='Model Performance Metrics (Logistic Regression)',
                      xaxis_title='Metrics',
                      yaxis_title='Percentage')
    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
    # # Display confusion matrix
    # st.write("Confusion Matrix:")
    # st.write(conf_matrix)
    return logistic_regression_accuracy*100


@st.cache_resource(ttl=43200, max_entries=1000)
def svm(X_train, X_test, y_train, y_test):
    # Train SVM with linear kernel
    svm_linear = SVC(kernel='linear', random_state=0)
    svm_linear.fit(X_train, y_train)
    # Perform cross-validation for linear SVM
    accuracies_linear = cross_val_score(estimator=svm_linear, X=X_train, y=y_train, cv=10)
    # Calculate mean accuracy and standard deviation for linear SVM
    mean_accuracy_linear = accuracies_linear.mean() * 100
    std_deviation_linear = accuracies_linear.std() * 100
    # Make predictions on test set for linear SVM
    pre_linear = svm_linear.predict(X_test)
    # Calculate accuracy on test set for linear SVM
    accuracy_linear = accuracy_score(pre_linear, y_test)
    # Calculate confusion matrix for linear SVM
    conf_matrix_linear = confusion_matrix(pre_linear, y_test)
    # Train SVM with RBF (Radial Basis Function) kernel
    svm_rbf = SVC(kernel='rbf', random_state=0)
    svm_rbf.fit(X_train, y_train)
    # Perform cross-validation for RBF SVM
    accuracies_rbf = cross_val_score(estimator=svm_rbf, X=X_train, y=y_train, cv=10)
    # Calculate mean accuracy and standard deviation for RBF SVM
    mean_accuracy_rbf = accuracies_rbf.mean() * 100
    std_deviation_rbf = accuracies_rbf.std() * 100
    # Make predictions on test set for RBF SVM
    pre_rbf = svm_rbf.predict(X_test)
    # Calculate accuracy on test set for RBF SVM
    accuracy_rbf = accuracy_score(pre_rbf, y_test)
    # st.write(accuracy_rbf , accuracy_linear)
    # Calculate confusion matrix for RBF SVM
    conf_matrix_rbf = confusion_matrix(pre_rbf, y_test)
    fig = go.Figure(data=[
        go.Bar(name='Linear SVM', x=['Mean Accuracy', 'Standard Deviation', 'Accuracy on Test Set'],
               y=[mean_accuracy_linear, std_deviation_linear, accuracy_linear * 100]),
        go.Bar(name='RBF SVM', x=['Mean Accuracy', 'Standard Deviation', 'Accuracy on Test Set'],
               y=[mean_accuracy_rbf, std_deviation_rbf, accuracy_rbf * 100])
    ])
    fig.update_layout(barmode='group', title='Performance Comparison: Linear vs RBF SVM',
                      xaxis_title='Metrics', yaxis_title='Percentage')
    st.plotly_chart(fig, use_container_width=True)
    return [accuracy_rbf*100, accuracy_linear*100]


@st.cache_resource(ttl=43200, max_entries=1000)
def KNN(X_train, X_test, y_train, y_test):
    scores = []
    for i in range(1, 10):
        knn = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
        knn.fit(X_train, y_train)
        pre3 = knn.predict(X_test)
        ans = accuracy_score(pre3, y_test)
        scores.append(round(100 * ans, 2))
    best_k = max(scores)
    # st.write("Best Accuracy:", best_k)

    # Create a line plot to visualize the accuracy for different values of k
    fig = go.Figure(data=go.Scatter(x=list(range(1, 10)), y=scores, mode='lines+markers'))
    fig.update_layout(title='Accuracy vs. Number of Neighbors (K)', xaxis_title='Number of Neighbors (K)',
                      yaxis_title='Accuracy (%)')
    st.plotly_chart(fig, use_container_width=True)
    return best_k


@st.cache_resource(ttl=43200, max_entries=1000)
def navebayes(X_train, X_test, y_train, y_test):
    gb = GaussianNB()
    gb.fit(X_train, y_train)
    accuracies = cross_val_score(estimator=gb, X=X_train, y=y_train, cv=10)
    mean_accuracy = accuracies.mean() * 100
    std_deviation = accuracies.std() * 100
    # Make predictions on the test set
    pre4 = gb.predict(X_test)
    # Calculate accuracy on the test set
    naive_bayes_accuracy = accuracy_score(pre4, y_test)
    # Calculate confusion matrix
    # Create a bar chart to compare accuracies
    fig = go.Figure(data=[
        go.Bar(name='Gaussian Naive Bayes', x=['Mean Accuracy', 'Standard Deviation', 'Accuracy on Test Set'],
               y=[mean_accuracy, std_deviation, naive_bayes_accuracy * 100])
    ])
    fig.update_layout(title='Model Performance Metrics: Gaussian Naive Bayes',
                      xaxis_title='Metrics', yaxis_title='Percentage')
    st.plotly_chart(fig, use_container_width=True)

    return naive_bayes_accuracy * 100


@st.cache_resource(ttl=43200, max_entries=1000)
def decison_tree(X_train, X_test, y_train, y_test):
    # Train the Decision Tree classifier
    dc = DecisionTreeClassifier(criterion='gini')
    dc.fit(X_train, y_train)
    # Perform cross-validation
    accuracies = cross_val_score(estimator=dc, X=X_train, y=y_train, cv=10)
    # Calculate mean accuracy and standard deviation
    mean_accuracy = accuracies.mean() * 100
    std_deviation = accuracies.std() * 100
    # Make predictions on the test set
    pre5 = dc.predict(X_test)
    # Calculate accuracy on the test set
    decision_tree_accuracy = accuracy_score(pre5, y_test)
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(pre5, y_test)
    # Display confusion matrix
    # st.write("Confusion Matrix:")
    # st.write(conf_matrix)

    fig = go.Figure(data=[
        go.Bar(name='Decision Tree Classifier', x=['Mean Accuracy', 'Standard Deviation', 'Accuracy on Test Set'],
               y=[mean_accuracy, std_deviation, decision_tree_accuracy * 100])
    ])
    fig.update_layout(title='Model Performance Metrics: Decision Tree Classifier',
                      xaxis_title='Metrics', yaxis_title='Percentage')
    st.plotly_chart(fig, use_container_width=True)
    return decision_tree_accuracy * 100


@st.cache_resource(ttl=43200, max_entries=1000)
def random_forest(X_train, X_test, y_train, y_test):
    accuracy_values = []
    mean_accuracy_values = []
    std_deviation_values = []

    for i in range(1, 100):
        rc = RandomForestClassifier(n_estimators=i, criterion='entropy', random_state=0)
        rc.fit(X_train, y_train)
        accuracies = cross_val_score(estimator=rc, X=X_train, y=y_train, cv=10)
        mean_accuracy = accuracies.mean() * 100
        std_deviation = accuracies.std() * 100
        pre6 = rc.predict(X_test)
        random_forest_accuracy = accuracy_score(pre6, y_test)
        conf_matrix = confusion_matrix(pre6, y_test)
        accuracy_values.append(random_forest_accuracy)
        mean_accuracy_values.append(mean_accuracy)
        std_deviation_values.append(std_deviation)

    fig = go.Figure(data=go.Scatter(x=list(range(1, 100)), y=accuracy_values, mode='lines'))
    fig.update_layout(title='Accuracy vs. Number of Estimators (Random Forest)',
                      xaxis_title='Number of Estimators', yaxis_title='Accuracy')
    st.plotly_chart(fig, use_container_width=True)
    return mode(accuracy_values)*100

def best_model(log_acc,svm_acc_rbf,svm_acc_lin,knn_acc,nav_acc,dec_acc,rand_acc):
    # Accuracy values for each model
    # st.write(log_acc,svm_acc_rbf,svm_acc_lin,knn_acc,nav_acc,dec_acc,rand_acc)
    models = ['Logistic Regression', 'SVM (Linear)', 'SVM (RBF)', 'KNN', 'Naive Bayes', 'Decision Trees',
              'Random Forest']
    accuracy_values = [log_acc,svm_acc_rbf,svm_acc_lin,knn_acc,nav_acc,dec_acc,rand_acc]

    # Create a bar chart to visualize the accuracy of each model
    fig = go.Figure(data=go.Bar(x=models, y=accuracy_values, marker=dict(color='skyblue')))
    fig.update_layout(title='Accuracy of Machine Learning Models',
                      xaxis_title='Models', yaxis_title='Accuracy')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    style.Style(title='Data Analysis', header=f"Model Performance Metrics")
    df = data.load_data("0")
    df_corr = data.corrleation_data(df, '0')
    X = df_corr[
        ['Polydipsia', 'sudden weight loss', 'partial paresis', 'Irritability', 'Polyphagia', 'Age', 'visual blurring']]
    y = df_corr['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    col1,col2 = st.columns(2)
    with col1:
        log_acc = logistic_regression(X_train, X_test, y_train, y_test)
    with col2:
        nav_acc = navebayes(X_train, X_test, y_train, y_test)
    col1, col2 = st.columns(2)
    with col1:
        svm_acc = svm(X_train, X_test, y_train, y_test)
        svm_acc_rbf = svm_acc[0]
        svm_acc_lin = svm_acc[1]

    with col2:
        knn_acc = KNN(X_train, X_test, y_train, y_test)
    col1, col2 = st.columns(2)
    with col1:
        rand_acc = random_forest(X_train, X_test, y_train, y_test)
    with col2:
        dec_acc = decison_tree(X_train, X_test, y_train, y_test)

    best_model(log_acc,svm_acc_rbf,svm_acc_lin,knn_acc,nav_acc,dec_acc,rand_acc)
