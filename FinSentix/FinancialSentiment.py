from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter import simpledialog
from sklearn.model_selection import train_test_split
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import re
from sklearn.preprocessing import StandardScaler
import pickle
import os
import pandas as pd
from sentence_transformers import SentenceTransformer #loading bert sentence model
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM #class for LSTM training
import os
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional,GRU
from keras.utils.np_utils import to_categorical
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer


# main = tkinter.Tk(): This line creates the main window of the graphical user interface using the Tk() method from the tkinter module. 
# The Tk() method initializes the main window for the application.
# main.title("FinSentix: Contextual Enrichment in Financial Script through Sentiment and Term Extraction"): This sets the title of the main window to 
# "FinSentix: Contextual Enrichment in Financial Script through Sentiment and Term Extraction". This title will appear in the title bar of the window.
# main.geometry("1000x650"): This sets the initial size of the main window to a width of 1000 pixels and a height of 650 pixels. 
# The format for specifying the size is "width x height".
main = tkinter.Tk()
main.title("FinSentix: Contextual Enrichment in Financial Script through Sentiment and Term Extraction") #designing main screen
main.geometry("1000x650")


# labels = ['Positive', 'Neutral', 'Negative']: This line creates a list called labels containing strings representing different sentiment categories. 
# In this case, the sentiment categories are 'Positive', 'Neutral', and 'Negative'. These labels are likely used for classification tasks or for 
# interpreting the sentiment predictions.
# global filename, X, Y, lstm_model, scaler: This line declares global variables (filename, X, Y, lstm_model, scaler) within the current scope. 
# This allows these variables to be accessed and modified from anywhere within the program.
# global X_train, X_test, y_train, y_test: Similar to the previous line, this line declares additional global variables (X_train, X_test, y_train, y_test) 
# within the current scope.
# global accuracy, precision, recall, fscore, dataset, bert: Declares global variables (accuracy, precision, recall, fscore, dataset, bert) within the current scope.
# stop_words = set(stopwords.words('english')): This line uses the stopwords corpus from the Natural Language Toolkit (NLTK) to create a set of English stopwords. 
# Stopwords are common words (such as "and", "the", "is") that are often filtered out during text preprocessing in natural language processing tasks.
# lemmatizer = WordNetLemmatizer(): This line initializes an instance of the WordNetLemmatizer class from the NLTK library. Lemmatization is a text 
# normalization technique that reduces words to their base or root form (e.g., "running" to "run").
# bert = SentenceTransformer('nli-distilroberta-base-v2'): Here, an instance of the SentenceTransformer class is created from the sentence_transformers library. 
# The SentenceTransformer class is used for generating embeddings (vector representations) of sentences using pre-trained transformer models. In this case, 
# the specific model used is 'nli-distilroberta-base-v2'.
labels = ['Positive', 'Neutral', 'Negative']
global filename, X, Y, lstm_model, scaler
global X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore, dataset, bert
stop_words = set(stopwords.words('english')) #initialize the stopwords
lemmatizer = WordNetLemmatizer() #initialize the lemmatizer
bert = SentenceTransformer('nli-distilroberta-base-v2') # initialize the model 

def cleanData(doc):
    tokens = doc.split()  # Splits the input document into individual words (tokens) based on whitespace by using the split() method.
    table = str.maketrans('', '', punctuation)  # Generates a translation table(intab:characters to be replaces, outtab: characters that replace with, deletetab: characters that have to be deleted)
    tokens = [w.translate(table) for w in tokens]  # Removes punctuation from each token using the translation table.
    tokens = [word for word in tokens if word.isalpha()]  # Removes tokens that contain non-alphabetic characters.
    tokens = [w for w in tokens if not w in stop_words]  # Removes tokens that are in the set of stopwords.
    tokens = [word for word in tokens if len(word) > 1]  # Removes tokens with a length less than or equal to 1.
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatizes each token (reduces it to its base form).
    tokens = ' '.join(tokens)  # Joins the processed tokens back into a string with spaces in between.
    return tokens  # Returns the cleaned and processed string.

def loadDataset():
    global filename, dataset  # Declares the filename and dataset variables as global, allowing them to be modified in this function.
    # Opens a file dialog window for the user to select a file from the 'Dataset' directory.
    filename = filedialog.askopenfilename(initialdir="Dataset")
    # Clears the text area (if named 'text') from the beginning (line 1) to the end (END).
    text.delete('1.0', END)
    # Inserts a message indicating the filename loaded into the text area.
    text.insert(END, filename + " loaded\n\n")
    # Reads a CSV file using Pandas (pd) from the selected filename.
    # Encoding is specified as 'iso-8859-1',(an encoding scheme commonly used to handle text files with characters from many languages, 
    # capable of representing a larger range of characters compared to the more common 'utf-8' encoding.) and 
    # '@' is used as the separator(The sep parameter defines the delimiter or separator used to differentiate between columns in the CSV file.).
    dataset = pd.read_csv(filename, encoding='iso-8859-1', sep="@")
    # Inserts a representation of the first few rows of the loaded dataset into the text area.
    text.insert(END, str(dataset.head()))
    # Counts unique values in the 'sentiments' column of the dataset using NumPy (np).
    label, count = np.unique(dataset['sentiments'], return_counts=True)
    # Prepares data for creating a bar plot of sentiment counts.
    height = count  # Represents the count of each sentiment type.
    bars = label  # Represents the different sentiment types.
    y_pos = np.arange(len(bars))  # Generates an array of positions for the bars on the x-axis.
    # Creates a bar plot using Matplotlib.
    plt.figure(figsize=(4, 3))  # Sets the size of the figure.
    plt.bar(y_pos, height)  # Plots the bars with their heights at specified positions.
    plt.xticks(y_pos, bars)  # Sets x-axis ticks as the different sentiment types.
    plt.xlabel("Sentiment Types")  # Sets the label for the x-axis.
    plt.ylabel("Count")  # Sets the label for the y-axis.
    plt.xticks(rotation=90)  # Rotates x-axis labels by 90 degrees for better readability.
    plt.tight_layout()  # Adjusts layout for better visualization.
    plt.show()  # Displays the created bar plot.


def processDataset():
    text.delete('1.0', END)  # Clears the content of the text area (if named 'text') from the beginning (line 1) to the end (END).
    global X, Y, dataset  # Declares global variables X, Y, and dataset to modify them within this function.
    X = []  # Initializes an empty list X to store processed data.
    Y = []  # Initializes an empty list Y to store labels or target values.
    dataset = dataset.values  # Converts the dataset DataFrame into a NumPy array for iteration.
    # Iterates through each row of the dataset using a for loop.
    for i in range(len(dataset)):
        news = dataset[i, 0]  # Retrieves the content of the first column (index 0) in each row.
        news = news.strip().lower()  # Strips leading and trailing whitespaces and converts the text to lowercase.
        news = cleanData(news)  # Calls the cleanData() function to preprocess the text.
        text.insert(END, str(news) + "\n")  # Inserts the processed text into the text area for display.


def bertEncoding():
    text.delete('1.0', END)  # Clears the content of the text area (if named 'text') from the beginning to the end.
    global X, Y, scaler, dataset  # Declares global variables X, Y, scaler, and dataset for modification.
    global X_train, X_test, y_train, y_test, bert  # Declares global variables for train-test splitting and BERT model.
    # Checks if the BERT-encoded data files already exist in the 'model' directory.
    if os.path.exists("model/bert.npy"):
        # If the files exist, load the BERT-encoded data and labels.
        X = np.load("model/bert.npy")
        Y = np.load("model/Y.npy")
    else:
        # If the BERT-encoded data files do not exist, create and save them.
        for i in range(len(dataset)):
            news = dataset[i, 0]  # Retrieves the text data from the dataset.
            news = news.strip().lower()  # Removes leading/trailing spaces and converts to lowercase.
            news = cleanData(news)  # Preprocesses the text using the cleanData() function.
            label = 0  # Default label value
            # Assigns labels based on the sentiment values in the dataset.
            if dataset[i, 1].strip().lower() == 'neutral':
                label = 1
            if dataset[i, 1].strip().lower() == 'negative':
                label = 2
            # Appends preprocessed text data and labels to X and Y lists, respectively.
            X.append(news)
            Y.append(label)
            print(news + " " + str(label) + " " + str(dataset[i, 1]))
        # Converts the Y list to a NumPy array for further processing.
        Y = np.asarray(Y)
        # Encodes text using the BERT model to generate embeddings (vectors).
        embeddings = bert.encode(X, convert_to_tensor=True)
        X = embeddings.numpy()
        # Saves the BERT-encoded data and labels to files in the 'model' directory.
        np.save("model/bert", X)
        np.save("model/Y", Y)
    # Randomizes the indices of the data to shuffle the dataset.
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    # Reorders the data and labels based on the shuffled indices.
    X = X[indices]
    Y = Y[indices]
    scaler = StandardScaler() #StandardScaler function in the scikit-learn library is used for standardizing features by removing the mean and scaling them to unit variance.
    X = scaler.fit_transform(X)  # Normalizes the dataset to values between 0 and 1.
    text.insert(END, "BERT NLP Encoding = " + str(X) + "\n\n")  # Inserts BERT-encoded data into the text area.
    X = np.reshape(X, (X.shape[0], 48, 16))  # Reshapes the data to match the required input shape for LSTM.
    Y = to_categorical(Y)  # Converts labels to categorical format (one-hot encoding).
    # Splits the dataset into training and testing sets with an 80-20 ratio.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    # Displays details about the train-test split in the text area.
    text.insert(END, "Dataset Train & Test Split Details\n\n")
    text.insert(END, "80% dataset records used for Training: " + str(X_train.shape[0]) + "\n")
    text.insert(END, "20% dataset records used for Testing: " + str(X_test.shape[0]) + "\n\n\n\n\n\n")
    text.update_idletasks()  # Updates the text area to display the inserted content.

    

#function to calculate accuracy, precision and other metrics and then plot confusion matrix
def prediction(predict, true_label, algorithm):
    global labels  # Accessing the global variable 'labels' for use in the function.
    # Calculating precision, recall, F1-score, and accuracy using respective functions from sklearn.
    p = precision_score(true_label, predict, average='macro') * 100
    r = recall_score(true_label, predict, average='macro') * 100
    f = f1_score(true_label, predict, average='macro') * 100
    a = accuracy_score(true_label, predict) * 100
    # Displaying the calculated metrics in the text area of the GUI.
    text.insert(END, algorithm + ' Accuracy  : ' + str(a) + "\n")
    text.insert(END, algorithm + ' Precision : ' + str(p) + "\n")
    text.insert(END, algorithm + ' Recall    : ' + str(r) + "\n")
    text.insert(END, algorithm + ' FScore    : ' + str(f) + "\n\n\n")
    # Generating a confusion matrix using the confusion_matrix function from sklearn.
    conf_matrix = confusion_matrix(true_label, predict)
    # Plotting a heatmap of the confusion matrix using seaborn and matplotlib.
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g") #annot  determines whether the cell values should be annotated or not, 
    #cmap  specifies the color map used to colorize the heatmap in this case viridis ranges from yellow to green to blue, fmt gives the format for annotation in cells of heat map, g stands for general format
    ax.set_ylim([0, len(labels)])  # Adjusting the heatmap height to match the number of labels.
    plt.title(algorithm + " Confusion matrix")  # Setting the title of the plot.
    plt.ylabel('True class')  # Labeling the y-axis of the heatmap.
    plt.xlabel('Predicted class')  # Labeling the x-axis of the heatmap.
    plt.show()  # Displaying the heatmap.


def trainLSTM():
    text.delete('1.0', END)  # Clears the content of the text area.
    # Declaring global variables to be used and modified in the function.
    global X, Y, scaler, dataset, lstm_model
    global X_train, X_test, y_train, y_test
    lstm_model = Sequential()  # Creating a Sequential model for the LSTM network.
    # Adding layers to the LSTM model.
    lstm_model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)) #An LSTM layer with 100 memory units is added to the model,
    # input_shape parameter defines the input shape of the data, return_sequences=True indicates the layer should return sequences of outputs.
    lstm_model.add(Dropout(0.2)) # A dropout layer is added with a dropout rate of 0.2, randomly deactivating 20% of the neurons to prevent overfitting.
    # Two Bidirectional GRU layers are included in the model architecture, Bidirectional GRU (Gated Recurrent Unit) layers are a type of recurrent neural network 
    # (RNN) architecture that processes input data in both forward and backward directions. They combine two separate recurrent networks (GRUs in this case) into 
    # one, capturing information from both past and future contexts simultaneously. A dropout layer with a 0.2 rate follows the GRU layer.
    lstm_model.add(Bidirectional(GRU(80, return_sequences=True)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Bidirectional(GRU(64)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(100, activation='relu'))  # A dense (fully connected) layer with 100 units and Rectified Linear Unit (ReLU) activation function is added to introduce non-linearity.
    lstm_model.add(Dense(y_train.shape[1], activation='softmax'))  #Output layer employs a softmax activation function suitable for multi-class classification, converting model outputs into class probabilities.
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # The model is compiled with the Adam optimizer, 
    # categorical cross-entropy loss function (suitable for multi-class classification), and accuracy as the evaluation metric during training.
    # Check if the model weights file exists. If not, train the model and save the weights.
    if not os.path.exists("model/lstm_weights.hdf5"):
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose=1, save_best_only=True) # Creating a checkpoint to save the best model weights during training.
        hist = lstm_model.fit(X_train, y_train, batch_size=64, epochs=30, validation_data=(X_test, y_test),
                               callbacks=[model_check_point], verbose=1) # Training the LSTM model with specified parameters.
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)  # Saving the training history to a file.
        f.close()
    else:
        lstm_model = load_model("model/lstm_weights.hdf5") # If the model weights file exists, load the model with these weights.
    predict = lstm_model.predict(X_test)  # Predicting using the trained LSTM model.
    predict = np.argmax(predict, axis=1)  # Getting the indices of the maximum values.
    y_test1 = np.argmax(y_test, axis=1)  # Converting one-hot encoded labels to categorical labels.
    prediction(predict, y_test1, "LSTM") # Using the 'prediction()' function to display evaluation metrics and confusion matrix for the LSTM model.

        
def predict():
    text.delete('1.0', END)  # Clears the Text widget content
    global bert, lstm_model, scaler
    filename = filedialog.askopenfilename(initialdir="Dataset")  # Opens a file dialog to select a dataset
    data = pd.read_csv(filename, encoding='iso-8859-1')  # Reads the selected CSV file into a pandas DataFrame
    temp = data.values  # Extracts the values from the DataFrame
    test = []
    # Preprocesses and prepares the data for prediction
    for i in range(len(temp)):
        news = str(temp[i]).strip().lower()  # Converts data to lowercase and removes leading/trailing whitespaces
        news = cleanData(news)  # Cleans and processes the textual data
        test.append(news)  # Appends the preprocessed data to the 'test' list
    test = np.asarray(test)  # Converts 'test' list to a NumPy array
    test = bert.encode(test, convert_to_tensor=True)  # Encodes the preprocessed data using BERT Sentence Transformer, convert it into a tensor(Tensor 
    #a multi-dimensional array that can represent higher-dimensional data efficiently. Tensors generalize scalars, vectors, and matrices to higher dimensions)
    test = test.numpy()  # Converts the encoded data to a NumPy array
    test = scaler.transform(test)  # Normalizes the dataset to a range between 0 and 1
    test = np.reshape(test, (test.shape[0], 48, 16))  # Reshapes the data for compatibility with the LSTM model
    predict = lstm_model.predict(test)  # Uses the LSTM model to make predictions on the test dataset
    # Analyzes and displays the predictions along with extracted data from the dataset
    for i in range(len(predict)):
        matches = re.findall('\d+\.\d+', str(temp[i]))  # Extracts numerical patterns from the data
        percentages = [str(match) for match in matches]  # Stores extracted percentages
        prices = re.findall(r'[\d\.\d]+', str(temp[i]))  # Extracts prices from the data
        price = []
        for k in range(len(prices)):
            if prices[k] not in percentages and prices[k] != ".":  # Filters out percentage values from prices
                price.append(prices[k])  # Stores non-percentage values in 'price' list
        pred = np.argmax(predict[i])  # Determines the predicted sentiment class index
        score = np.amax(predict[i])  # Retrieves the maximum score from the prediction
        # Inserts various extracted information and predictions into the Text widget for display
        text.insert(END, "Financial News = " + str(temp[i]) + "\n")
        text.insert(END, "Extracted % = " + str(percentages) + "\n")
        text.insert(END, "Extracted Prices = " + str(price) + "\n")
        text.insert(END, "Predicted Sentiments = " + str(labels[pred]) + "\n")
        text.insert(END, "Sentiments Score = " + str(score) + "\n\n")


#main.destroy() is a method used on a Tkinter window object (main in this case). It closes the window, terminating the GUI application.
def close():
    main.destroy()

# Font definition for the title
font = ('times', 16, 'bold')
# Creating a Label widget for the title in the main window
title = Label(main, text='FinSentix: Contextual Enrichment in Financial Script through Sentiment and Term Extraction', justify=LEFT)
# Configuring the appearance of the title Label widget
title.config(bg='lavender blush', fg='DarkOrchid1')  # Setting background and foreground colors
title.config(font=font)  # Applying the defined 'font' to the title Label
title.config(height=3, width=120)  # Setting the height and width of the Label
# Placing the title Label at specific coordinates within the window
title.place(x=100, y=5)  # Specifies the position of the Label using x and y coordinates
# Packing the title Label (this is not needed if using place or grid)
title.pack()  # This line will not have any effect if place is used, as they are conflicting geometry managers


# Font definition for the button text
font1 = ('times', 13, 'bold')
# Creating a Button widget for uploading financial news dataset in the main window
uploadButton = Button(main, text="Upload Financial News Dataset", command=loadDataset)
# Placing the uploadButton at specific coordinates within the window
uploadButton.place(x=10, y=100)  # Specifies the position of the Button using x and y coordinates
# Configuring the font style of the uploadButton text
uploadButton.config(font=font1)  # Applying the defined 'font1' to the text of the Button


# Font definition for the button text
font1 = ('times', 13, 'bold')
# Creating a Button widget for preprocessing the dataset in the main window
processButton = Button(main, text="Preprocess Dataset", command=processDataset)
# Placing the processButton at specific coordinates within the window
processButton.place(x=330, y=100)  # Specifies the position of the Button using x and y coordinates
# Configuring the font style of the processButton text
processButton.config(font=font1)  # Applying the defined 'font1' to the text of the Button


# Font definition for the button text
font1 = ('times', 13, 'bold')
# Creating a Button widget for converting text to BERT NLP encoding in the main window
bertButton = Button(main, text="Text to Bert NLP Encoding", command=bertEncoding)
# Placing the bertButton at specific coordinates within the window
bertButton.place(x=620, y=100)  # Specifies the position of the Button using x and y coordinates
# Configuring the font style of the bertButton text
bertButton.config(font=font1)  # Applying the defined 'font1' to the text of the Button


# Font definition for the button text
font1 = ('times', 13, 'bold')
# Creating a Button widget for training the LSTM algorithm in the main window
lstmButton = Button(main, text="Train LSTM Algorithm", command=trainLSTM)
# Placing the lstmButton at specific coordinates within the window
lstmButton.place(x=10, y=150)  # Specifies the position of the Button using x and y coordinates
# Configuring the font style of the lstmButton text
lstmButton.config(font=font1)  # Applying the defined 'font1' to the text of the Button


# Font definition for the button text
font1 = ('times', 13, 'bold')
# Creating a Button widget for predicting financial sentiments in the main window
predictButton = Button(main, text="Predict Financial Sentiments", command=predict)
# Placing the predictButton at specific coordinates within the window
predictButton.place(x=330, y=150)  # Specifies the position of the Button using x and y coordinates
# Configuring the font style of the predictButton text
predictButton.config(font=font1)  # Applying the defined 'font1' to the text of the Button


# Font definition for the button text
font1 = ('times', 13, 'bold')
# Creating a Button widget for exiting the application in the main window
exitButton = Button(main, text="Exit", command=close)
# Placing the exitButton at specific coordinates within the window
exitButton.place(x=620, y=150)  # Specifies the position of the Button using x and y coordinates
# Configuring the font style of the exitButton text
exitButton.config(font=font1)  # Applying the defined 'font1' to the text of the Button


# Font definition for the Text widget
font1 = ('times', 12, 'bold')
# Creating a Text widget (multiline text field) in the main window
text = Text(main, height=22, width=140)
# Creating a Scrollbar for the Text widget
scroll = Scrollbar(text)
# Configuring the Text widget to use the Scrollbar
text.configure(yscrollcommand=scroll.set)
# Placing the Text widget at specific coordinates within the window
text.place(x=10, y=200)  # Specifies the position of the Text widget using x and y coordinates
# Configuring the font style of the text displayed within the Text widget
text.config(font=font1)  # Applying the defined 'font1' to the text in the Text widget
# Configuring the background color of the main window
main.config(bg='light coral')  # Setting the background color of the main window
# Starting the main event loop to display the GUI
main.mainloop()  # Enters the main event loop to handle user inputs and events

