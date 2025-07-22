
Open the web application in your browser (usually at http://localhost:8501).

Use the sliders and dropdown menus in the sidebar to enter the employee's details.

Click the "Predict Salary Class" button.

The predicted income class and the confidence score will be displayed on the main page.

📂 Project Structure
Here is an overview of the key files in this project:

.
├── 📄 app.py                    # The main Python script for the Streamlit application
├── 📦 best_model_pipeline.pkl   # The pre-trained machine learning model pipeline
├── 📦 encoders.pkl              # The saved encoders for categorical features
├── 📄 requirements.txt           # A list of Python libraries required for the project
└── 📄 README.md                 # This file

📊 Data
The model was trained on the Adult Census Income dataset, which is a publicly available dataset from the UCI Machine Learning Repository. It contains 14 attributes used to predict whether a person's income exceeds $50K a year.

📜 License
This project is licensed under the MIT License. See the LICENSE file for more details.
