from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import uuid

app = Flask(__name__)

# Function to analyze student performance
def analyze_student_performance(data):
    # Basic statistical analysis
    basic_stats = data.describe()
    
    # Correlation analysis
    correlation_matrix = data.corr()
    
    # Perform t-test to compare high vs low study hours
    median_study = data['study_hours'].median()
    high_study = data[data['study_hours'] > median_study]['final_grade']
    low_study = data[data['study_hours'] <= median_study]['final_grade']
    t_stat, p_value = stats.ttest_ind(high_study, low_study)
    
    # Linear regression
    X = data[['study_hours', 'sleep_hours', 'prev_gpa']]
    y = data['final_grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    results = {
        'basic_stats': basic_stats,
        'correlation_matrix': correlation_matrix,
        'ttest_results': {
            't_statistic': t_stat,
            'p_value': p_value
        },
        'regression_coefficients': dict(zip(X.columns, model.coef_)),
        'model_r2': model.score(X_test, y_test)
    }
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[0, 0])
    axes[0, 0].set_title('Correlation Heatmap')
    
    sns.scatterplot(data=data, x='study_hours', y='final_grade', ax=axes[0, 1])
    axes[0, 1].set_title('Study Hours vs. Final Grade')
    
    sns.histplot(data=data, x='final_grade', bins=20, ax=axes[1, 0])
    axes[1, 0].set_title('Grade Distribution')
    
    data['study_category'] = pd.qcut(data['study_hours'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    sns.boxplot(data=data, x='study_category', y='final_grade', ax=axes[1, 1])
    axes[1, 1].set_title('Grades by Study Time Category')
    
    plt.tight_layout()
    
    # Save the plot as a unique image file
    image_filename = f'static/images/{uuid.uuid4()}.png'
    plt.savefig(image_filename)
    plt.close()
    
    return results, image_filename

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        
        # Check if file is an Excel or CSV file
        if file and (file.filename.endswith(".xlsx") or file.filename.endswith(".csv")):
            if file.filename.endswith(".xlsx"):
                data = pd.read_excel(file)
            else:
                data = pd.read_csv(file)
            
            # Ensure required columns are present
            required_columns = {'study_hours', 'sleep_hours', 'prev_gpa', 'final_grade'}
            if not required_columns.issubset(data.columns):
                return "Uploaded file must contain 'study_hours', 'sleep_hours', 'prev_gpa', and 'final_grade' columns."
            
            # Run the analysis
            results, image_path = analyze_student_performance(data)
            
            # Send the image and results to the template
            return render_template("results.html", results=results, image_path=image_path)
        else:
            return "Please upload an Excel or CSV file."

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
