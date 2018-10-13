import sys
import nltk
import pandas as pd
import numpy as np

from vocabulary import Vocabulary

#When the last function needed after which the data will be ready to be fed as input to the network, we will create
#a function that is going to call all of the other functions/
class DataPreprocessing:

    """This class is responsible for preprocessing the raw data that are given as inputself.
    @input_file: the complete path to the xls file containing the raw data"""

    def __init__(self,input, sheet_name):
        self.input_file = input
        self.xl = None
        self.raw_data = None
        self.np_data = None
        self.processed_data = None
        self.sheet_name = sheet_name

    def readInput(self):
        self.xl = pd.read_excel(self.input_file, sheet_name = self.sheet_name)
        self.raw_data = self.xl
        return self.xl

    def preprocess(self):
        """This function tokenizes the sentences, normalizes the grades,makes all words of the essay lowercase, and handles unknown words"""
        print ("Creating a vocabulary...")
        vocabulary = Vocabulary(self.input_file)
        vocabulary.readInput()
        print ("Getting Vocabulary as a set...")
        voc_set,_ = vocabulary.vocabulary_as_set()
        print (type(voc_set))
        print (voc_set)
        print ("Got that set...")
        #Tokenize the sentences into words
        print ("Tokenizing...")
        self.raw_data['essay'] = self.raw_data['essay'].apply(nltk.word_tokenize)
        print ("Tokenized. Making every word lowercase...")
        # Normalize the grades to [0,1] space and then lower case every word in the corpus
        for i in range(self.raw_data.shape[0]):
            if self.raw_data['essay_set'].iloc[i] == 1:
                self.raw_data['domain1_score'].iloc[i] /= 12
            elif self.raw_data['essay_set'].iloc[i] == 2:
                self.raw_data['domain1_score'].iloc[i] /= 6
            elif self.raw_data['essay_set'].iloc[i] == 3:
                self.raw_data['domain1_score'].iloc[i] /= 3
            elif self.raw_data['essay_set'].iloc[i] == 4:
                self.raw_data['domain1_score'].iloc[i] /= 3
            elif self.raw_data['essay_set'].iloc[i] == 5:
                self.raw_data['domain1_score'].iloc[i] /= 4
            elif self.raw_data['essay_set'].iloc[i] == 6:
                self.raw_data['domain1_score'].iloc[i] /= 4
            elif self.raw_data['essay_set'].iloc[i] == 7:
                self.raw_data['domain1_score'].iloc[i] /= 30
            else:
                self.raw_data['domain1_score'].iloc[i] /= 60

            for j in range(len(self.raw_data['essay'].iloc[i])):
                self.raw_data['essay'].iloc[i][j]= self.raw_data['essay'].iloc[i][j].lower()
                if self.raw_data['essay'].iloc[i][j] not in voc_set:
                    self.raw_data['essay'].iloc[i][j] = '*' #Ampersand is the special character that replaces all words that are not part of the vocabulary
        print ("Finished with data preprocessing...Getting back to aes model")
        return self.raw_data

    def get_processed_numpy_data(self):
        """This function converts the xls file into a numpy array"""
        self.np_data = self.raw_data.as_matrix()

        return self.np_data
