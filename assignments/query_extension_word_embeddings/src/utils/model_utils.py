from argparse import ArgumentParser
from typing import List, Tuple

import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors

from .logging_utils import get_logger

logger = get_logger(__name__)


def load_gensim_model(model_name: str) -> KeyedVectors:
    """
    Loads a Gensim model with the given model name.

    Parameters:
        model_name (str): The name of the Gensim model to load.

    Returns:
        KeyedVectors: The loaded Gensim model.

    Raises:
        ValueError: If the model is not found.

    """
    logger.info(f" Attempting to load model: {model_name} from gensim...")
    try:
        model = api.load(model_name)
        logger.info(f"Model: {model_name} has been loaded successfully!")
        return model
    except ValueError:
        raise ValueError(f"Model not found: {model_name}")
    except Exception as e:
        raise Exception(f"An error occurred while loading the model: {e}")


def get_word_embeddings(
    model: KeyedVectors, parser: ArgumentParser, word: str
) -> List[Tuple[str, float]]:
    """
    Retrieves word embeddings for a given word from a pre-trained word embedding model.

    Parameters:
        model (KeyedVectors): The pre-trained word embedding model.
        parser (ArgumentParser): The argument parser object.
        word (str): The word for which to retrieve word embeddings.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing the most similar words and their similarity scores.
    """
    logger.info(f"Attempting to get word embeddings for: {word} from model...")
    try:
        most_similar = model.most_similar(word, topn=10)
        logger.info(f"Word embeddings for: {word} have been retrieved successfully!")
        return most_similar
    except KeyError:
        parser.exit(
            1, message="[MODEL_ERROR] Input query not found in model vocabulary"
        )


def check_word_in_model_vocabulary(model: KeyedVectors, word: str) -> bool:
    """
    Check if a word is present in the vocabulary of a word embedding model.

    Parameters:
    - model (KeyedVectors): The word embedding model.
    - word (str): The word to check.

    Returns:
    - True if the word is in the model's vocabulary, False otherwise.
    """
    return word in model.vocab
