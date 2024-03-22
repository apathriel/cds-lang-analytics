

import argparse
from utils.utilities import convert_string_to_lower_case

def get_cli_args():
    parser = instantiate_parser()
    parser.add_argument('-a', '--artist', nargs='+', type=str, help='Artist name', required=True)
    parser.add_argument('-q', '--query', type=str, help='Query word', required=True)
    parser.add_argument('-m', '--model', type=str, help='Word embedding model from gensim', required=False, default='glove-wiki-gigaword-50')
    parser.add_argument('-s', '--save', action='store_true', help='Saves the query results to a csv file', required=False)
    parser.add_argument('-o', '--output', type=str, help='Output file name', required=False, default='query_results.csv')

    args = parser.parse_args()
    args.query = convert_string_to_lower_case(args.query)
    return parser, args

def instantiate_parser():
    return argparse.ArgumentParser(
        prog="corpus_query",
        description='This script is used to query a dataset of songs by artist. The query will examine a specified concept, this query is extended through word embeddings.',
        epilog='Thanks for using %(prog)s!'
        )