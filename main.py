import argparse
import sys

from core.constants import *
from core.texts_estimator import TextsEstimator


def check_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, help='Path to directory with text files to evaluate',
                        required=True)
    parser.add_argument('-pp', '--prompt_path', type=str, help='Text evaluation prompt path', required=True)
    parser.add_argument('-mp', '--models_list_path', type=str, help="Path to file that contains list of "
                                                                    "LLM's names that will be used to evaluate texts:",
                        required=True)
    parser.add_argument('-o', '--output_filename', type=str, help='The name of file that will be '
                                                                  'generated as the result of process',
                        required=False, default="estimated_texts.csv")
    parser.add_argument('-ms', '--max_size', type=int, help='Maximum size of file that will be evaluated '
                                                            '(in kilobytes)',
                        required=False, default=None)
    parsed = parser.parse_args(args)
    return parsed.data_path, parsed.prompt_path, parsed.models_list_path, parsed.output_filename, parsed.max_size


if __name__ == '__main__':
    data_path, prompt_path, models_list_path, output_filename, max_size = check_arguments(sys.argv[1:])

    with open(models_list_path, 'r') as file:
        models = file.read().splitlines()

    texts_evaluator = TextsEstimator(api_key=API_KEY, prompt_file_link=prompt_path,
                                     models=models, output_filename=output_filename)
    texts_evaluator.evaluate_texts(data_path=data_path, max_size=max_size)
