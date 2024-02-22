#!/usr/bin/env python3

"""
Author: Gabriel Høst Andersen
Date: 22-02-2024
"""

import pandas as pd
import spacy
import os
import chardet
import re

def clean_text(text):
    """
    This function uses a regular expression to remove all occurrences of tags in text.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text with all tags removed.
    """
    return re.sub(r'<[^>]+>', '', text)

def open_text_file(filepath):
    """
    Opens and reads a text file, returning its contents as a string.

    The function assumed UTF-8 encoding, if it fails, attempts to determine the encoding using chardet module, try loading text file with new encoding.

    Args:
        filepath (str): The path to the file to open.

    Returns:
        str: The contents of the file as a string.

    Raises:
        UnicodeDecodeError: If the file cannot be decoded using UTF-8 encoding.
    """
    try:
        with open (filepath, "r", encoding="utf-8") as file:
            return clean_text(file.read())
    except UnicodeDecodeError:
        with open (filepath, "rb") as file:
            get_encoding = chardet.detect(file.read())
        try:
            with open (filepath, "r", encoding=get_encoding['encoding']) as file:
                return clean_text(file.read())
        except:
            print("ERROR")

def calculate_relative_frequency(document, count, per_how_many_words=10000, remove_punctuation=True):
    """
    Calculates the relative frequency of a certain type of token (pos) in document loaded through a spaCy model, per a certain number of words.

    Args:
        document (List) (tokens processed by spaCy model): The document (doc) class to analyze, represented as a list of tokens.
        count (int): The number of occurrences of the token type in the document.
        per_how_many_words (int, optional): The number of words per which the relative frequency is calculated.
        remove_punctuation (bool, optional): Whether to remove punctuation tokens from the document before calculating the relative frequency.

    Returns:
        float: The relative frequency of the token in the document, rounded to two decimal places.
    """
    if remove_punctuation:
        return round(sum(1 for token in document if not token.is_punct) / len(document) * per_how_many_words, 2)
    else:
        return round(count / len(document) * per_how_many_words, 2)

def process_folders(input_path, output_path, model, remove_punctuation=False):
    """
    Processes all text files in the given input folder and its subfolders, and writes the results to CSV files in the designated output folder.

    This function iterates over all subfolders in the input folder. For each text file in each subfolder, it opens the file,
    processes the text using the given spaCy model, and calculates various statistics about the text.

    Args:
        input_path (str): The path to the input folder.
        output_path (str): The path to the output folder.
        model (spacy.lang): The spaCy language model to use for NLP.
        remove_punctuation (bool, optional): Whether to remove punctuation tokens from the documents before calculating the relative frequency.
    """
    input_folders = os.listdir(input_path)

    for sub_folder in input_folders:
        print(f"[SYSTEM] Processing subfolder {sub_folder}")
        input_sub_folder_path = os.path.join(input_path, sub_folder)
        input_sub_folders = os.listdir(input_sub_folder_path)
        df_list = []  # create list for storing dataframes, append each, then concatenate all at once

        for file in input_sub_folders:
            print(f"[SYSTEM] Processing file: {file}")
            text = open_text_file(os.path.join(input_sub_folder_path, file))
            doc = model(text)
            values_dict = count_occurances_type(doc, remove_punctuation) | count_occurances_named_entity(doc)
            create_df_row = pd.DataFrame({
                'Filename': [file], 
                'RelFreq NOUN': [values_dict['noun_count']],
                'RelFreq VERB':[values_dict['verb_count']],
                'RelFreq ADJ': [values_dict['adj_count']],
                'RelFreq ADV': [values_dict['adv_count']],
                'No. Unique PER':[values_dict['PERSON']],
                'No. Unique LOC': [values_dict['LOC']],
                'No. Unique ORG': [values_dict['ORG']]
                })
            df_list.append(create_df_row)  # append the DataFrame a list

        df = pd.concat(df_list, ignore_index=True)  # concatenate all the DataFrames at once
        df.to_csv(f"{os.path.join(output_path, sub_folder + '_table' + '.csv')}", index=False)
        print('[SYSTEM] Subfolder has been processed')
    return print('[SYSTEM] All folders have been processed')

def count_occurances_named_entity(document):
    """
    Counts the number of unique occurrences of certain types of named entities in a document.

    Args:
        document (spacy.doc): The document to analyze, represented as a spaCy Doc object.

    Returns:
        dict: A dictionary where the keys are named entity labels ('PERSON', 'ORG', 'LOC') and the values are the counts of unique named entities of each type.

    Raises:
        TypeError: If a named entity's label is not a string.
        KeyError: If a named entity's label is not one of the specified types.
    """
    PERSON, ORG, LOC = 'PERSON', 'ORG', 'LOC'
    entity_counts = {PERSON: set(), ORG: set(), LOC: set()}

    for ent in document.ents:
        try:
            match ent.label_:
                case 'PERSON':
                    entity_counts[PERSON].add(ent.text)
                case 'ORG':
                    entity_counts[ORG].add(ent.text)
                case 'LOC':
                    entity_counts[LOC].add(ent.text)
                case _:
                    continue
        except TypeError:
            print("[ERROR] Value is of wrong type")
        except KeyError:
            print("[ERROR] Key does not exist :(")
        
    #dictionary comprehension - create new dict applying len function to each key
    return {label: len(count) or 0 for label, count in entity_counts.items()}


def count_occurances_type(document, remove_punctuation=False):
    """
    Counts the occurrences of certain types of tokens in a spacy document object.

    Args:
        document (spacy.doc): The document to analyze, represented as a spaCy Doc object.

    Returns:
        dict: A dictionary where the keys are token types ('noun', 'verb', 'adj', 'adv') and the values are the counts of tokens of each type.
    """
    NOUN, VERB, ADJ, ADV = 'noun_count', 'verb_count', 'adj_count', 'adv_count'
    pos_counts = {
        NOUN: 0,
        VERB: 0,
        ADJ: 0,
        ADV: 0
    }

    for token in document:
        try:
            match token.pos_:
                case "NOUN":
                    pos_counts[NOUN] += 1
                case "VERB":
                    pos_counts[VERB] += 1
                case "ADJ":
                    pos_counts[ADJ] += 1
                case "ADV":
                    pos_counts[ADV] += 1
                case _: 
                    continue
        except TypeError:
            print("[ERROR] Value is not a number")
        except KeyError:
            print("[ERROR] Key does not exist :(")

    return {key: calculate_relative_frequency(document, value, 10000, remove_punctuation) for key, value in pos_counts.items()}

if __name__ == "__main__":
    # script is intended to be run from the assignment_01 directory
    nlp = spacy.load("en_core_web_md")
    input_folder_path = os.path.join("data", "input")
    output_folder_path = os.path.join("data", "output")
    
    process_folders(input_folder_path, output_folder_path, nlp)