import argparse
import numpy as np

def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("validation_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix
def log_sum_exp(X):  #suppose X is a vector
    m = np.max(X)
    exp = 0
    for x in X:
        exp += np.exp(x-m)
    logexp = np.log(exp)
    logexp = logexp + m
    return logexp

def forwardbackward(seq, loginit, logtrans, logemit):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix
    
    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    L = len(seq)
    M = len(loginit)

    # Initialize log_alpha and fill it in
    log_alpha = np.zeros((L, M))
    for t in range(L):
        xt = seq[t]
        for j in range(M):  # in t round
            if (t + 1) == 1:
                log_alpha_tj = loginit[j] + logemit[j, words.index(xt)]  # a number
            else:
                v_alpha = log_alpha[t - 1] + logtrans[:, j]  # log_alpha -a vector with j cols; col j of trans
                # v being a vector of j cols
                log_alpha_tj = logemit[j, words.index(xt)] + log_sum_exp(v_alpha)  # a number
            log_alpha[t, j] = log_alpha_tj

    # Initialize log_beta and fill it in
    log_beta = np.zeros((L, M))
    for t in range(L - 1, -1, -1):
        for j in range(M):  # in t round
            if t == L - 1:
                log_beta_tj = np.log(1)  # a number
            else:
                xt1 = seq[t + 1]
                v_beta = logemit[:, words.index(xt1)] + logtrans[j, :] + log_beta[
                    t + 1]  # log_alpha -a vector with j cols; col j of trans
                # v being a vector of j cols
                log_beta_tj = log_sum_exp(v_beta)  # a number

            log_beta[t, j] = log_beta_tj
    #alpha = np.exp(log_alpha)
    #beta = np.exp(log_beta)

    # Compute the predicted tags for the sequence
    pred_py = np.zeros((L, M))
    y_pred = []
    for t in range(L):
        #pred_py[t] = alpha[t] * beta[t]  # M(j) cols
        pred_py[t] = log_alpha[t] + log_beta[t]
        y_pred.append(tags[np.argmax(pred_py[t])])

    # Compute the log-probability of the sequence
    log_p = log_sum_exp(log_alpha[L-1])

    # Return the predicted tags and the log-probability
    return y_pred, log_p
    pass
    

    
if __name__ == "__main__":
    # Get the input data
    validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = get_inputs()
    x_validation = []  # N rows, T cols
    y_validation = []  # N rows, T cols
    for i in validation_data:
        xi = []
        yi = []
        for t in i:
            xi.append(t[0])
            yi.append(t[1])
        x_validation.append(xi)
        y_validation.append(yi)

    tags = list(tags_to_indices.keys())
    words = list(words_to_indices.keys())

    # For each sequence, run forward_backward to get the predicted tags and
    # the log-probability of that sequence.
    N = len(validation_data)
    # T = validation_data.shape[1]
    y_validation_pred = []
    log_p_validation = []
    for xi in x_validation:
        yi_pred, log_pi = forwardbackward(xi, np.log(hmminit), np.log(hmmtrans), np.log(hmmemit))
        y_validation_pred.append(yi_pred)  # N rows, L cols
        log_p_validation.append(log_pi)  # N rows,
    log_p_validation = np.array(log_p_validation)

    # Compute the average log-likelihood and the accuracy. The average log-likelihood
    # is just the average of the log-likelihood over all sequences. The accuracy is
    # the total number of correct tags across all sequences divided by the total number
    # of tags across all sequences.
    avg_log_p_validation = np.mean(log_p_validation)

    ls_acc = []  # np.zeros((len(y_validation),max([len(i) for i in y_validation])))
    for i in range(N):
        T = len(y_validation[i])
        for t in range(T):
            if y_validation_pred[i][t] == y_validation[i][t]:
                ls_acc.append(1)  # [i,t] = 1
            else:
                ls_acc.append(0)
    accuracy = sum(ls_acc) / len(ls_acc)  # (acc_matrix.shape[0]*acc_matrix.shape[1])

    # Save the output files
    with open(predicted_file, 'w') as f:
        for i in range(len(x_validation)):
            for t in range(len(x_validation[i])):
                f.write(str(x_validation[i][t]) + '\t' + str(y_validation_pred[i][t] + '\n'))
            f.write('\n')
        f.write('\n')

    with open(metric_file, 'w') as f:
        f.write('Average Log-Likelihood: ' + str(avg_log_p_validation))
        f.write('\n')
        f.write('Accuracy: ' + str(accuracy))

    pass