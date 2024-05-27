# News Classification with BERT

This project demonstrates how to use BERT for classifying news articles into various categories.

## Project Structure

news-classification
│
├── data
│ └── 2024_4.csv
│
├── notebooks
│ └── news_classification.ipynb
│
├── src
│ ├── dataset.py
│ ├── train.py
│ ├── evaluate.py
│ └── inference.py
│
├── results
│ └── training_logs.png
│
├── .gitignore
├── requirements.txt
└── README.md


## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/news-classification.git
   cd news-classification
2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
3. **Add your Dataset**
    Place your dataset (2024_4.csv) in the data directory.

Running the Project

Training the Model
Navigate to the src directory and run the training script:

sh
Copy code
cd src
python train.py
Evaluating the Model
After training, you can evaluate the model:

sh
Copy code
python evaluate.py
Inference
You can use the trained model to classify new headlines:

sh
Copy code
python inference.py
Output

Training and Evaluation
The model will be trained using the provided dataset. The training script will output logs indicating the progress of the training process. After training, you can evaluate the model using the evaluation script, which will print out the accuracy and a detailed classification report:

yaml
Copy code
Accuracy: 0.8532967032967033
Classification Report: 
               precision    recall  f1-score   support

     Business       0.82      0.92      0.87       688
Entertainment       0.90      0.94      0.92       672
    Headlines       0.81      0.45      0.58       951
       Health       0.94      0.96      0.95       620
      Science       0.91      0.92      0.92       550
       Sports       0.81      0.96      0.88       699
   Technology       0.94      0.96      0.95       763
    Worldwide       0.70      0.88      0.78       517

     accuracy                           0.85      5460
    macro avg       0.85      0.87      0.86      5460
 weighted avg       0.85      0.85      0.84      5460
Inference
When you run the inference script, you can input a new headline and get the predicted category. For example:

sh
Copy code
python inference.py
Output:

csharp
Copy code
The headline "New breakthrough in AI technology" is classified as: Technology