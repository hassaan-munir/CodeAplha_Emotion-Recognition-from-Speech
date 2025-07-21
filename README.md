# Emotion Recognition From Speech

## Project Overview

In this project, you are requested to train a **deep learning model** to recognize human feelings (such as happy, sad, angry) based on audio sampling of speech sounds. The model will use enhanced machine learning to be able to correctly identify the emotional condition based on the sound of the voice.

## Dataset

In training and testing the model, the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset** was employed. This is a complete data that includes recordings of actors uttering different sentences in different moods.

You can download the dataset from Kaggle:  
[RAVDESS Emotional Speech Speech Emotion Recognition](https://www.kaggle.com/api/v1/datasets/download/uwrfkaggler/ravdess-emotional-speech-audio)

## Methodology

The development process followed a standard Machine Learning and Deep Learning pipeline:

1.  **Data Acquisition & Preprocessing:**
      * The RAVDESS dataset was loaded and processed.
      * Audio files were padded to a consistent length to ensure uniform input for feature extraction.
2.  **Feature Extraction:**
      * **Mel-Frequency Cepstral Coefficients (MFCCs)** were extracted from each audio file. MFCCs are powerful features that represent the unique spectral characteristics of speech relevant to emotion.
      * Each audio file was transformed into a fixed-size MFCC feature vector.
3.  **Data Preparation:**
      * Emotion labels (e.g., 'happy', 'angry') were converted into a numerical, **One-Hot Encoded** format suitable for deep learning classification.
      * The dataset was split into training and testing sets (75% training, 25% testing) to ensure robust model evaluation on unseen data.
4.  **Model Architecture & Training:**
      * A **Convolutional Neural Network (CNN)** was designed and implemented. CNNs are well-suited for pattern recognition in sequential data like audio features.
      * The model was trained using the Adam optimizer and categorical cross-entropy loss.
      * **ModelCheckpoint** and **EarlyStopping** callbacks were utilized to save the best performing model and prevent overfitting during training.
5.  **Model Evaluation:**
      * The trained CNN model's performance was rigorously evaluated on the test set.
      * Key metrics such as **Accuracy, Precision, Recall, and F1-Score** were calculated.
      * A **Confusion Matrix** was generated to provide a visual breakdown of the model's classification performance across different emotion categories.

## Key Results

The trained CNN model gave a great **~[94.7%] accuracy** on the test set. This exhibits the high potential of the model in identifying different emotions of human beings through audio of speech. The classification report also revealed high precision, recall, and F1-scores in most classes of emotions, which imply good performance of the model.

## Technologies Used

  * **Python**: The primary programming language. 
  * **Librosa**: For audio processing and feature extraction (MFCCs).
  * **Soundfile**: For reading and writing audio files.
  * **TensorFlow / Keras**: For building, training, and evaluating the deep learning (CNN) model.
  * **Scikit-learn**: For data splitting, label encoding, and comprehensive model evaluation metrics.
  * **NumPy**: For efficient numerical operations and array manipulation.
  * **Pandas**: For data handling and structuring (though less prominent in this specific notebook's snippet, it's generally used).
  * **Matplotlib**: For plotting training history and confusion matrices.
  * **Seaborn**: For enhanced statistical data visualization.
  * **Google Colab**: Used as the development environment, leveraging its cloud resources and GPU capabilities.

## How to Run

To run this project:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hassaan-munir/CodeAlpha_Speech_Emotion_Recognition.git
    cd CodeAlpha_Speech_Emotion_Recognition
    ```
    
2.  **Download the Dataset:**
      * [ Click Here ](https://www.kaggle.com/api/v1/datasets/download/uwrfkaggler/ravdess-emotional-speech-audio) to Download Data Set
      * Download the dataset (`RAVDESS_Speech_Emotion_Recognition.zip`).
      * Extract the contents. You should find an `archive` folder inside.
      * Upload the `archive` folder (containing `Actor_01` to `Actor_24` subfolders, possibly nested within `audio_speech_actors_01-24`) to your Google Drive, preferably in a `Colab Notebooks` folder or similar, matching the `DATA_PATH` in the notebook.
3.  **Open in Google Colab:**
      * Upload the `Emotion Recognition from Speech.ipynb` notebook to Google Colab.
      * Ensure your Google Drive is mounted (`from google.colab import drive; drive.mount('/content/drive')`).
      * Verify and adjust the `DATA_PATH` variable in the notebook to point to your `archive` folder in Google Drive.
4.  **Install Libraries & Run:**
      * Run the first cell to install all required libraries (`!pip install ...`).
      * Run all subsequent cells sequentially to load data, extract features, build, train, and evaluate the model.

## Connect with Me

**Muhammad Hassaan Munir** [LinkedIn Profile](https://www.linkedin.com/in/muhammad-hassaan-munir-79b5b2327/)
