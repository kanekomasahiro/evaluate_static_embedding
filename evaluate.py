import argparse
import copy
import itertools
import numpy as np

from gensim.models import KeyedVectors
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding', type=str, required=True)
    parser.add_argument('--benchmarks', type=str, required=True,
                        help='')

    args = parser.parse_args()

    return args


def calculate_matrix_cosine_similarity(matrix1, matrix2):
    return np.dot(matrix1, matrix2.T) / (np.linalg.norm(matrix1, axis=1) * np.linalg.norm(matrix2, axis=1)).reshape(-1, 1)


def calculate_vector_cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def load_embedding_with_gensim(embedding_name):
    if embedding_name.endswith('bin'):
        binary = True
    else:
        binary = False
    embedding = KeyedVectors.load_word2vec_format(embedding_name, binary=binary)

    return embedding


def flat_list_of_lists(inputs):
    return itertools.chain.from_iterable(inputs)


def get_most_similar_word(target_vector, words, embedding):
    target_vector = target_vector.reshape(1, -1)
    cosine_similarities = calculate_matrix_cosine_similarity(embedding[words], target_vector)
    max_index = np.argmax(cosine_similarities)
    predicted_word = words[max_index]

    return predicted_word


def evaluate_word_analogy(data, embedding):
    correct_num = 0
    total_num = 0
    unk_num = 0
    unk_words = []
    data_words = [word for word in set(flat_list_of_lists(data)) if word in embedding]

    for word1, word2, word3, word4 in data:
        if word1 in embedding and word2 in embedding and word3 in embedding and word4 in embedding:
            total_num += 1
            words = copy.deepcopy(data_words)
            words.remove(word1)
            words.remove(word2)
            words.remove(word3)
            target_vector = embedding[word2] - embedding[word1] + embedding[word3]
            predicted_word = get_most_similar_word(target_vector, words, embedding)
            if predicted_word == word4:
                correct_num += 1
        else:
            unk_num += 1
            if word1 not in embedding:
                unk_words.append(word1)
            if word2 not in embedding:
                unk_words.append(word2)
            if word3 not in embedding:
                unk_words.append(word3)
            if word4 not in embedding:
                unk_words.append(word4)

    return correct_num / total_num, unk_num, list(set(unk_words))


def evaluate_semantic_similarity(data, embedding):
    cosine_similarities = []
    human_ratings = []
    unk_num = 0
    unk_words = []

    for word1, word2, human_rating in data:
        if word1 in embedding and word2 in embedding:
            cosine_similarities.append(calculate_vector_cosine_similarity(embedding[word1], embedding[word2]))
            human_ratings.append(float(human_rating))
        else:
            unk_num += 1
            if word1 not in embedding:
                unk_words.append(word1)
            if word2 not in embedding:
                unk_words.append(word2)

    correlation, p_value = stats.spearmanr(cosine_similarities, human_ratings)

    return correlation, p_value, unk_num, list(set(unk_words))


def main(args):

    embedding = load_embedding_with_gensim(args.embedding)
    semantic_similarity_benchmarks = ['men', 'simlex', 'mturk', 'mc',
                                      'ws', 'rw', 'rg', 'scws',
                                      'behavior']
    word_analogy_benchmarks = ['google', 'msr']

    for benchmark in args.benchmarks.split(','):
        print(benchmark)
        with open(f'benchmarks/{benchmark}.txt') as f:
            if benchmark in semantic_similarity_benchmarks:
                data = [line.strip().split() for line in f]
                correlation, p_value, unk_num, unk_words = evaluate_semantic_similarity(data, embedding)
                print('correlation:', format(correlation * 100, '.1f'))
                print('p value:', p_value)
            elif benchmark in word_analogy_benchmarks:
                data = [line.strip().split() for line in f
                        if len(line.split()) == 4]
                accuracy, unk_num, unk_words = evaluate_word_analogy(data, embedding)
                print('accuracy:', format(accuracy * 100, '.1f'))


        print('data size:', len(data))
        print('skip number:', unk_num)
        #print('unk words:', unk_words)


if __name__ == "__main__":
    args = parse_args()
    main(args)
