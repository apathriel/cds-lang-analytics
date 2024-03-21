import gensim.downloader as api

def load_gensim_model(model_name: str):
    print(f"[SYSTEM] Attempting to load model: {model_name} from gensim...")
    try:
        model = api.load(model_name)
        print(f"[SYSTEM] Model: {model_name} has been loaded successfully!")
        return model
    except ValueError:
        print(f"Model not found: {model_name}")
        return

def get_word_embeddings(model, parser, word: str):
    print(f"[SYSTEM] Attempting to get word embeddings for: {word} from model...")
    try:
        most_similar = model.most_similar(word, topn=10)
        print(f"[SYSTEM] Word embeddings for: {word} have been retrieved successfully!")
        return most_similar
    except KeyError:
        parser.exit(1, message="[PARSER_ERROR] Input query not found in model vocabulary")

def check_word_in_model_vocabulary(model, word: str) -> bool:
    return word in model.vocab