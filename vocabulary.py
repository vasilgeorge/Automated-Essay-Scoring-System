import nltk
import pandas as pd
from nltk.corpus import stopwords
import sys
from collections import Counter


class Vocabulary:

    def __init__(self, input):
        self.input = input
        self.df = None
        self.sorted_words = None
        self.voc_set = None
        self.vocabulary_list = None

    def readInput(self):
        #Read the excel file and transform it into a DataFrame
        xl = pd.read_excel(self.input, sheet_name = 'trainin_set')
        self.df = xl
        print ("Input read")
        return xl

    def getTotalWords(self):
        for i in range(self.df.shape[0]):
            tokenized_essay = nltk.word_tokenize(self.df['essay'].iloc[i])
            if i == 0:
                total_words = tokenized_essay
            else:
                total_words += tokenized_essay

        for i in range(len(total_words)):
            total_words[i] = total_words[i].lower()

        return total_words

    def countInFile(self):
        #We should ignore words starting with @ as well as stopwords that do not contriute to the deeper meaning of the text
        """The vocabulary that is created in this function is of size 4000. However, the original vocabulary is going to be of size 40001,
        including the special character for the unknown words"""

        ignore = {'@', 't', 's', "'s'", "n't", 'caps1-', 'caps1.v',"caps2't"}
        for i in range(100):
            ignore.add("num"+str(i))
            ignore.add("caps"+str(i))
            ignore.add("month"+str(i))
            ignore.add("organization"+str(i))
            ignore.add("person"+str(i))
            ignore.add("date"+str(i))
            ignore.add("time"+str(i))
            ignore.add("city"+str(i))
            ignore.add("percent"+str(i))
            ignore.add("state"+str(i))
            ignore.add("location"+str(i))
            ignore.add("dr"+str(i))
            ignore.add("email"+str(i))
        total_words = self.getTotalWords()
        #We use isalpha() if we want to get rid of the punctuation
        self.sorted_words = pd.DataFrame(Counter(word for word in total_words if word not in ignore).most_common(4000), columns = ['Word', 'Count']).set_index('Word')
        pd.set_option('display.max_rows', len(self.sorted_words))
        pd.reset_option('display.max_rows')

        return self.sorted_words

    def vocabulary_as_set(self):
        print("Getting total words..")
        total_words = self.getTotalWords()
        ignore = {'@','t', 's', "'s'", "n't", 'caps1-', 'caps1.v',"caps2't"}
        for i in range(100):
            ignore.add("num"+str(i))
            ignore.add("caps"+str(i))
            ignore.add("month"+str(i))
            ignore.add("organization"+str(i))
            ignore.add("person"+str(i))
            ignore.add("date"+str(i))
            ignore.add("time"+str(i))
            ignore.add("city"+str(i))
            ignore.add("percent"+str(i))
            ignore.add("state"+str(i))
            ignore.add("location"+str(i))
            ignore.add("dr"+str(i))
            ignore.add("email"+str(i))
        print("Creating voc list")
        self.vocabulary_list = [word for word,_ in Counter(total_words).most_common(4000) if word not in ignore]
        self.voc_set = set(self.vocabulary_list)

        return self.voc_set, self.vocabulary_list

    def map_word_to_int(self):
        print("Mapping words to ints")
        _, voc_list = self.vocabulary_as_set()
        string_to_int = dict((c,i) for i, c in enumerate(voc_list, 1)) #Indexing starts from 1
        int_to_string = dict((i,c) for i, c in enumerate(voc_list, 1))
        #The star '*' is the special character into which every unknown word is mapped.
        string_to_int.update({'*':len(string_to_int)-1})
        int_to_string.update({len(string_to_int)-1:'*'})
        return string_to_int, int_to_string
