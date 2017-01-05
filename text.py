#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Helper functions to process text.

@author: amagrabi

"""

import numpy as np
import pandas as pd

# Language processing
import re
import spacy
import enchant
from inflection import pluralize
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases
from unidecode import unidecode
from html import unescape

import os
DIR_BASE = os.getcwd()
DIR_STATIC = os.path.join(DIR_BASE, 'static')

# Load language models
spacy = spacy.load('en')
spellcheck = enchant.Dict("en_US")

    
def encode_filenames(str_list):
    '''Transforms list of strings to valid filenames.
    
    Args:
        str_list: List of input names.
        
    Returns:
        List of transformed names.
        
    '''                             
    str_list = [item.replace("/", " ") for item in str_list]
    str_list = [item.replace("\\", " ") for item in str_list]
    str_list = [item.replace("&", "and") for item in str_list]
    str_list = [item.replace("\\", " ") for item in str_list]
    str_list = [item.lower() for item in str_list]
    str_list = [item.strip() for item in str_list]
    str_list = [unidecode(item) for item in str_list]
    return str_list

     
def decode_filenames(str_list):
    '''Transforms encoded list of strings to original names.
    
    Note: Can only be an approximation of the original filename.
    
    Args:
        str_list: List of encoded names.
        
    Returns:
        List of decoded names.
        
    '''
    str_list = [item.replace("and", "&") for item in str_list]
    str_list = [item.replace("_", " ") for item in str_list]
    str_list = [item.upper() for item in str_list]
    return str_list


def spellchecker(word):
    '''Corrects the spelling of a word.
    
    Spellchecker is based on a pre-loaded enchant-model ('spellcheck'-variable).
    
    Args:
        word: Word to be spellchecked.
        
    Returns:
        Spellcorrected word.
        
    '''
    # If the word is not empty
    if word != '':
        # If the word is not grammatical
        if not spellcheck.check(word):
            # Get suggestions
            word_suggestions = spellcheck.suggest(word) 
            # If there are in fact suggestions, return the first one
            if word_suggestions != []:
                return word_suggestions[0]
    # Otherwise: return uncorrected word
    return word
       
    
def similar_words(name, lexicon):
    '''Replaces words in a product name with the most similar words from a lexicon.
    
    Similarity is quantified via a pre-trained word2vec model 
    (e.g. based on GoogleNews data) loaded as word2vec_google at the start
    of the module.
    
    Args:
        name: Input product name.
        lexicon: lexicon to...
        
    Returns:
        Product name with similar words.
        
    '''
    name_bow = str.split(name)
    
    similarity = pd.DataFrame(np.zeros((len(lexicon), len(name_bow))), 
                              columns = name_bow) 
    similarity = similarity.set_index([lexicon], drop=True, inplace=False)
    
    for i in range(len(name_bow)):
        if name_bow[i] in word2vec_google.vocab:
            for j in range(len(lexicon)):
                if lexicon[j] in word2vec_google.vocab:
                    similarity.ix[j, i] = word2vec_google.similarity(name_bow[i], 
                                                                  lexicon[j])
    
    name_bow_similar = name_bow
    for i in range(len(name_bow)):
        sorted_col = similarity.ix[:,i].sort_values(ascending=False)
        if sorted_col.values[0] > 0.3:
            name_bow_similar[i] = sorted_col.index.tolist()[0]
    
    return ' '.join(name_bow_similar)
    

def clean_name(name, lexicon=[], min_word_len=3, spellchecking=True):
    '''Transforms an input name to a standardized form suited for analysis.
    
    Args:
        name: Input name.
        lexicon: List of strings. If passed, the cleaned name will only contain words that occur in the lexicon.
        min_word_len: Minimum length that words need to have to be returned in the cleaned name.
        spellchecking: Flag to enable spellchecking on each word (based on enchant-model).
        
    Returns:
        Cleaned name as string.
        
    '''
    # convert html entities
    name = unescape(name)
    
    # convert special characters
    name = name.replace('/',' ')
    name = name.replace('\\',' ')
    name = name.replace('"',' ')
    name = name.replace('“',' ')
    name = name.replace('\'',' ')
    name = name.replace('+',' ')
    name = name.replace('&',' ')
    
    # deal with hyphens
    name = re.sub('(-){2,}', '-', name)
    name = re.sub('( -)', ' ', name)
    name = re.sub('(- )', ' ', name)
    name = re.sub(r'\b-\b', '_', name)
    
    # get the tokens using spaCy
    name_parsed = spacy(name)

    # remove puctuation
    name_parsed = [word for word in name_parsed if not word.is_punct]

    # remove stopwords
    name_parsed = [word for word in name_parsed if not word.is_stop]

    # remove space
    name_parsed = [word for word in name_parsed if not word.is_space]

    # remove numbers
    name_parsed = [word for word in name_parsed if not word.like_num]

    # spellcheck words ()
    # name_parsed -> name_bow
    name_bow = [str(word) for word in name_parsed]
    # spellcheck words
    if spellchecking:
        name_bow_tmp = []
        for word in name_bow:
            # special case: hyphenated words
            if '_' in word:
                subword_corrected = []
                for subword in word.replace('_',' ').split():
                    subword_corrected.append(spellchecker(subword))
                name_bow_tmp.append('_'.join(subword_corrected))
            else:
                name_bow_tmp.append(spellchecker(word))
        name_bow = name_bow_tmp
    # name_bow -> name_parsed (back again)
    name_parsed = spacy(' '.join(name_bow))
    
    # lemmatize/stemming (also lowers strings and converts to list)
    name_parsed = [word.lemma_ for word in name_parsed]

    # Reduce to unique words?
#    name_bow = list(set(name_bow))
   
    # If lexicon is passed, remove words that are not in the lexicon
    if len(lexicon) > 0:
        bad_words = list(set(name_parsed) - set(lexicon))
        if bad_words:
            for word in bad_words:
                name_parsed.remove(word)
            
    # Remove short words -> has lead to worse prediction performance
    name_parsed = [word for word in name_parsed if len(word) >= min_word_len]
    
    return ' '.join(name_parsed)
            

def names_to_lexicon(names_list, min_occurrence=5):
    '''Constructs a word lexicon from an input text (as a list of strings).
    
    Args:
        names_list: Input list of strings.
        min_occurrence: Minimum number of occurrences words have to reach to get added to the lexicon.
        
    Returns:
        Tuple of the lexicon and the raw lexicon (=lexicon without word occurrence restriction).
        
    '''
    print('--- Building Lexicon ---')
    
    lexicon_raw = []
    for name in names_list:
        name_bow = str.split(clean_name(name))
        [lexicon_raw.append(x) for x in name_bow]

    # Delete low-n    
    df_lexicon_raw = pd.DataFrame({'lexicon_raw': lexicon_raw})
          
    counts = df_lexicon_raw['lexicon_raw'].value_counts()
    remove = []
    for word in lexicon_raw:
        if counts[word] <= min_occurrence:
            remove.append(word)
    lexicon = list(set(lexicon_raw) - set(remove))
    
    return (lexicon, lexicon_raw)
    
    
def text_to_phrases(names, min_count=4, threshold=1):
    '''...
    
    Args:
        ...
        
    Returns:
        ...
        
    '''
    names_list = [name.split() for name in names]
                  
    phrase_model = Phrases(names_list, min_count=min_count, threshold=threshold)
    
    names_phrased = []
    for name in list(phrase_model[names_list]):
        names_phrased.append(' '.join(name))
    
    return (names_phrased, phrase_model)
    
    
def phrasify_name(name, phrase_model):
    '''...
    
    Args:
        ...
        
    Returns:
        ...
        
    '''
    # str2bow -> phrasify -> bow2str
    return ' '.join(phrase_model[name.split()])
    
    
def dephrasify_name(name):
    '''...
    
    Args:
        ...
        
    Returns:
        ...
        
    '''
    return name.replace('_',' ')


def match_categories(cats_target, cats_prediction):
    '''Check whether a category corresponds to another category in a given category list.
    
    Args:
        ...
        
    Returns:
        ...
        
    '''
    cats_target_original = cats_target

    cats_target = [clean_name(cat) for cat in cats_target]
    cats_prediction = [clean_name(cat) for cat in cats_prediction]

    prediction_matches = []
    # For all predicted categories
    for i in range(len(cats_prediction)):
        # Test against each target category
        for j in range(len(cats_target)):
            if cats_prediction[i] == cats_target[j]:
                prediction_matches.append((i,j))
            # If target category does not match, test each word in each cat
            else:
                for word_prediction in cats_prediction[i].split():
                    for word_target in cats_target[j].split():
                        if word_prediction == word_target:
                            prediction_matches.append((i,j))
    
    # Find all predicted categories that correspond to a target category (unique)
    ind_predict_matches = [int(i[0]) for i in prediction_matches]
    seen = set()
    seen_add = seen.add
    ind_predict_matches = [x for x in ind_predict_matches if not (x in seen or seen_add(x))]                    
        
    # If there are multiple matching target categories, associate target category with the first match    
    matches_predict = []
    for ind_match in ind_predict_matches:
        ind_target = [prediction_matches[i][1] for i, v in enumerate(prediction_matches) if v[0] == ind_match][0]       
        matches_predict.append((ind_match, ind_target))
        
    # If there are multiple matching prediction categories, associate target category with the first match
    seen = set()
    matches_unique = [item for item in matches_predict if item[1] not in seen and not seen.add(item[1])]

    # Compute set of predicted categories (matched categories, with names of target categories)
    cats_prediction_new = []
    for match in matches_unique:
        cats_prediction_new.append(cats_target_original[match[1]])

    ind_matches_unique = [int(i[0]) for i in matches_unique]
    
    return (cats_prediction_new, ind_matches_unique)
        

def change_textfile(textfile, changefile):
    '''Transforms strings in a textfile based on a dictionary saved in another textfile.
    
    Args:
        textfile: Input textfile.
        changefile: Textfile of a dictionary indicating string transformations.
        
    '''
    changelist = {}
    with open(changefile, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
           (key, val) = line.split(': ')
           changelist[key] = val
        f.close()

    pattern = re.compile(r'\b(' + '|'.join(changelist.keys()) + r')\b')

    with open(textfile, 'r') as f:
        lines = f.readlines()
        f.close()
        
    with open(textfile, 'w') as f:
        for i, line in enumerate(lines):
            lines[i] = pattern.sub(lambda x: changelist[x.group()], line)
        f.writelines(lines)
        f.close()
        