'''Set of functions to support the creation of more comlicated test data.'''
import random
import string

import pandas as pd
import numpy as np

np.random.seed(42)


def generate_fake_vocabulary(voc_size=100):
    fake_vocab = []
    for _ in range(voc_size):
        word_length = random.randint(3, 8)
        fake_word = ''.join(random.choices(string.ascii_lowercase,
                                           k=word_length))
        fake_vocab.append(fake_word)
    return fake_vocab


def generate_fake_text(vocabulary, min_word_count=30, max_word_count=300):
    word_count = random.randint(min_word_count, max_word_count)
    fake_text = random.choices(vocabulary, k=word_count)
    fake_text = ' '.join(fake_text)
    return fake_text


def create_mock_data(num_rows):
    fake_vocab1 = generate_fake_vocabulary(voc_size=200)
    fake_vocab2 = generate_fake_vocabulary(voc_size=100)
    fake_vocab3 = generate_fake_vocabulary(voc_size=300)

    state_list = ["NY", "NH", "CA", "FL", "NM", "NC", "ID"]
    letter_list = ["A", "B", "C", "D"]
    feature_1 = [1, 2, 3, 4, 5]
    feature_2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    feature_3 = [True, False]

    data = {
        'id': [x for x in range(num_rows)],
        'state': [random.choice(state_list) for _ in range(num_rows)],
        'letter': [random.choice(letter_list) for _ in range(num_rows)],
        'feature_1': [random.choice(feature_1) for _ in range(num_rows)],
        'feature_2': [random.choice(feature_2) for _ in range(num_rows)],
        'feature_3': [random.choice(feature_3) for _ in range(num_rows)],
        'text_1': [generate_fake_text(fake_vocab1) for _ in range(num_rows)],
        'text_2': [generate_fake_text(fake_vocab2) for _ in range(num_rows)],
        'text_3': [generate_fake_text(fake_vocab3) for _ in range(num_rows)]
    }
    return data


def main():
    num_rows = 127
    output_file_path = "pipeline_test_df.pkl"
    data = create_mock_data(num_rows)
    data_df = pd.DataFrame(data)
    data_df.to_pickle(output_file_path)


if __name__ == '__main__':
    main()
