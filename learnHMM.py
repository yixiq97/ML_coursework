import argparse
import numpy as np


def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmmprior", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()

    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    return train_data, words_to_indices, tags_to_indices, args.hmmprior, args.hmmemit, args.hmmtrans


if __name__ == "__main__":
    # Collect the input data
    train_data, words_to_indices, tags_to_indices, hmmprior, hmmemit, hmmtrans = get_inputs()
    x_train = []  # N rows, T cols
    y_train = []  # N rows, T cols
    for i in train_data:
        xi = []
        yi = []
        for t in i:
            xi.append(t[0])
            yi.append(t[1])
        x_train.append(xi)
        y_train.append(yi)

    tags = list(tags_to_indices.keys())
    words = list(words_to_indices.keys())
    N = len(train_data)  # x_train.shape(0)
    # T = max([len(i) for i in train_data]) #x_train.shape(1)

    # Initialize the initial, emission, and transition matrices
    init = np.zeros(len(tags))  # same index as tags
    emit = np.zeros((len(tags), len(words)))
    trans = np.zeros((len(tags), len(tags)))  # same index as tags

    # Increment the matrices
    # init
    for yij in [yi[0] for yi in y_train]:
        for j in range(len(tags)):  # from 0 to
            if yij == tags[j]:
                init[j] += 1

    # emit
    for i in range(N):  # row
        T = len(train_data[i])
        for t in range(T):  # col
            emit[tags.index(y_train[i][t]), words.index(x_train[i][t])] += 1

    # trans
    for row in y_train:
        T = len(row)
        for t in range(T - 1):
            trans[tags.index(row[t]), tags.index(row[t + 1])] += 1

    # Add a pseudocount
    init = init + 1
    emit = emit + 1
    trans = trans + 1

    init = init / sum(init)
    emit = emit / np.sum(emit, axis=1).reshape(-1, 1)
    trans = trans / np.sum(trans, axis=1).reshape(-1, 1)

    # Save your matrices to the output files --- the reference solution uses
    # np.savetxt (specify delimiter="\t" for the matrices)
    np.savetxt(hmmprior, init, delimiter=" ")
    np.savetxt(hmmemit, emit, delimiter=" ")
    np.savetxt(hmmtrans, trans, delimiter=" ")

    pass