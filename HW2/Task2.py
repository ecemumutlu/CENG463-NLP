import re
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

def read_sentences_from_file_to_megadoc():
    folder_path = "./Task2"
    megadoc = list()
    documents_dict = dict()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_number = file.split('.')[0]
            file_path = os.path.join(root, file)
            
            # there are some latin characters that caused problem in reading the file in utf-8 format
            # so I used latin-1 encoding instead
            file = open(file_path, 'r', encoding='latin-1') 
            xml_file = file.read()
            file.close()
            
             # match the sentence tag and the text inside
            pattern = re.compile(r'<sentence.*?>([^<]*)</sentence>')
            matches = re.findall(pattern, xml_file)
            
            sentences = ""
            documents_dict[file_number] = list()
            for match in matches:
                sentence = match.strip().replace('\n', ' ') # remove new line characters
                documents_dict[file_number].append(sentence) # [sentence, sentence, ...]
                sentences += sentence + " "
            megadoc.append(sentences) # [doc, doc, ...]
            
    
    return megadoc, documents_dict



def main(file_name):
    megadoc, documents_dict = read_sentences_from_file_to_megadoc()
    
    # Create a tfidf vectorizer with the stop words removed and max_df set to 0.8
    # Max df is used to remove words that appear in more than 80% of the documents, which I do not find very informative
    tfidfvect = TfidfVectorizer(stop_words='english', max_df=0.8 )
    
    # Use the megadoc to fit the tfidf vectorizer
    # This will create a tfidf vectorizer with the vocabulary of the megadoc.
    # Fit_transform is used to fit the vectorizer and then transform the megadoc into a tfidf vectorized matrix
    tfidf_vectorized_values = tfidfvect.fit_transform(megadoc)
    
    # My document dictionary  is indexed according to the file name without the file extension part
    # So here I extract the file name first
    file_name = file_name.split('.')[0] #file_name should not include the folder path, ie. 06.3.xml
    
    doc = documents_dict[file_name]
    
    # Transform the document into a tfidf vectorized matrix
    tfidf_for_each_doc = tfidfvect.transform(doc)
    
    # Cosine similarity is calculated between the sentence and the document it resides in, 
    # where each sentence is vectorized according to the megadoc vectorizer
    cosine_sim_between_doc_and_megadoc = cosine_similarity(tfidf_for_each_doc, tfidf_for_each_doc)
    
    # Get the top 5 sentences by sorting first and then getting the top 5 indices
    top_5_indices = cosine_sim_between_doc_and_megadoc.argsort(axis=None)[::-1][:5]
    top_5_indices_2d = np.unravel_index(top_5_indices, cosine_sim_between_doc_and_megadoc.shape)
    
    doc_summary = [doc[i] for i in top_5_indices_2d[0]]
    
    for sentence in doc_summary:
        print(sentence)

if __name__ == "__main__":
    main(sys.argv[1])