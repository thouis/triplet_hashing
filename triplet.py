from collections import defaultdict
from scipy.sparse import coo_matrix
import numpy as np

def index_group_by_label(labels):
    """ compute indices into argument grouped by value
    """
    idx_by_label = defaultdict(list)
    for idx, l in enumerate(labels):
        idx_by_label[l].append(idx)
    return idx_by_label


def index_loop_over_triplets(labels):
    """ Generator for all indices of triplets (i1, i2, i3) such that
    labels[i1] == labels[i2] != labels[i3] and (i1 < i2)
    """
    idx_by_label = index_group_by_label(labels)

    for l1_l2, idx_l1_l2 in idx_by_label.items():
        for l3, idx_l3 in idx_by_label.items():
            if l1_l2 == l3:
                continue
            for i1 in idx_l1_l2:
                for i2 in idx_l1_l2:
                    if i1 >= i2:
                        continue
                    for i3 in idx_l3:
                        yield (i1, i2, i3)


def find_W(current_hashes, labels, triplet_loss):
    """ Find the weight matrix used to solve for next hash bits to append.
        (equation 8 of Zhuang et al.)

    Args:
        current_hashes: a list or array current hash values
            New hashes will be 2x these values + (0 or 1)
            NB: be sure representation has sufficient range.
        labels: list or array of true labels
        triplet_loss: function that can be called with arguments:
            (hash1, hash2, has3) where label1 == label2 != label3
            and returns the loss for that triple.

    Returns:
        new hashes approximately minimizing the triplet loss over
        all pairs.

    See section 3.1 of
    Fast Training of Triplet-based Deep Binary Embedding Networks
    Bohan Zhuang, Guosheng Lin, Chunhua Shen, Ian Reid
    http://arxiv.org/abs/1603.02844
    """

    # helper function for equation 6
    m_inv = np.matrix([[ 1,  1,  1,  1],
                       [ 1,  1, -1, -1],
                       [ 1, -1,  1, -1],
                       [ 1, -1, -1,  1]]).I

    def subloss(idx1, idx2, idx3):
        h1 = current_hashes[idx1]
        h2 = current_hashes[idx2]
        h3 = current_hashes[idx3]
        current_loss = triplet_loss(h1, h2, h3)
        # slightly different notation thann the paper:
        #   0 rather than -1
        lr_111 = triplet_loss(h1 * 2 + 1, h2 * 2 + 1, h3 * 2 + 1) - current_loss
        lr_110 = triplet_loss(h1 * 2 + 1, h2 * 2 + 1, h3 * 2 + 0) - current_loss
        lr_101 = triplet_loss(h1 * 2 + 1, h2 * 2 + 0, h3 * 2 + 1) - current_loss
        lr_100 = triplet_loss(h1 * 2 + 1, h2 * 2 + 0, h3 * 2 + 0) - current_loss
        return m_inv.dot([lr_111, lr_110, lr_101, lr_100]).A.ravel().tolist()

    # create W_ij - weights of interactions between all pairs
    ij_alphas = []
    # loop over all triples with label1 == label2 != label3
    for i, j, k in index_loop_over_triplets(labels):
        # see equations 5 & 6
        a_ii, a_ij, a_ik, a_jk = subloss(i, j, k)
        ij_alphas.extend([(a_ii, (i, i)),
                          (a_ij, (i, j)),
                          (a_ik, (i, k)),
                          (a_jk, (j, k))])

    alphas, ij_s = zip(*ij_alphas)
    a_i, a_j = zip(*ij_s)
    W = coo_matrix((alphas, (a_i, a_j)))
    return W


def compute_loss(hashes, labels, triplet_loss):
    """ compute the current loss for all triplets with the given hashes
    Args:
        (see find_W, below)
    Returns: sum of loss for each triplet
    """
    loss = 0.0
    for i, j, k in index_loop_over_triplets(labels):
        loss += triplet_loss(hashes[i], hashes[j], hashes[k])
    return loss


def find_next_bits(current_hashes, labels, triplet_loss):
    """ Find the next bit to append to each hash value to greedily minimze a triplet loss.


    Args:
        current_hashes: a list or array current hash values
            New hashes will be 2x these values + (0 or 1)
            NB: be sure representation has sufficient range.
        labels: list or array of true labels
        triplet_loss: function that can be called with arguments:
            (hash1, hash2, has3) where label1 == label2 != label3
            and returns the loss for that triple.

    Returns:
        new hashes approximately minimizing the triplet loss over
        all pairs.

    See section 3.1 of
    Fast Training of Triplet-based Deep Binary Embedding Networks
    Bohan Zhuang, Guosheng Lin, Chunhua Shen, Ian Reid
    http://arxiv.org/abs/1603.02844
    """

    old_loss = compute_loss(current_hashes, labels, triplet_loss)
    print("old", old_loss)

    print("finding W")
    W = find_W(current_hashes, labels, triplet_loss)

    # choose new hashes
    new_bits = [(idx % 2) for idx in range(len(current_hashes))]
    new_hashes = [h * 2 + o for h, o in zip(current_hashes, new_bits)]
    new_loss = compute_loss(new_hashes, labels, triplet_loss)

    # convert to 1/-1 vector
    bits_as_vec = np.matrix(new_bits) * 2 - 1
    print (W.todense())
    print("new", new_loss)
    print("delta", new_loss - old_loss)
    print("pred", bits_as_vec.dot(W.dot(bits_as_vec.T)))


def hamming(h1, h2):
    """ hamming distance between h1 and h2 """
    return bin(h1 ^ h2).count('1')


def hinged_triplet_loss_maker(bitlen):
    def hinged_triplet_loss(h1, h2, h3):
        # print(h1, h2, h3, "->", max(0, bitlen / 2.0 - (hamming(h1, h3) - hamming(h1, h2))))
        return max(0, bitlen / 2.0 - (hamming(h1, h3) - hamming(h1, h2)))
    return hinged_triplet_loss

if __name__ == '__main__':
    labels = ([0] * 3) + ([1] * 3) + ([3] * 5) + ([4] * 5)
    hashes = [0] * 3 + [1] * 3 + [5] * 5 + list(range(5))
    find_next_bits(hashes, labels, hinged_triplet_loss_maker(4))
