from collections import defaultdict
from scipy.sparse import coo_matrix
import numpy as np
import pymaxflow
from random import choice, shuffle
import itertools

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


def find_W(current_hashes, labels, triplet_loss, hashlen):
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
        hashlen: number of bits in current hashes

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
        current_loss = triplet_loss(h1, h2, h3, hashlen)
        # slightly different notation than the paper:
        #   0 rather than -1
        lr_111 = triplet_loss(h1 * 2 + 1, h2 * 2 + 1, h3 * 2 + 1, hashlen + 1) - current_loss
        lr_110 = triplet_loss(h1 * 2 + 1, h2 * 2 + 1, h3 * 2 + 0, hashlen + 1) - current_loss
        lr_101 = triplet_loss(h1 * 2 + 1, h2 * 2 + 0, h3 * 2 + 1, hashlen + 1) - current_loss
        lr_100 = triplet_loss(h1 * 2 + 1, h2 * 2 + 0, h3 * 2 + 0, hashlen + 1) - current_loss
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


def compute_loss(hashes, labels, triplet_loss, hashlen):
    """ compute the current loss for all triplets with the given hashes
    Args:
        (see find_W, below)
    Returns: sum of loss for each triplet
    """
    loss = 0.0
    for i, j, k in index_loop_over_triplets(labels):
        loss += triplet_loss(hashes[i], hashes[j], hashes[k], hashlen)
    return loss


def generate_submodular(W):
    """Generates a sequence of (possibly overlapping) submodular submatrices of W,
       until all rows have been sampled at least once.

    Args:
        W: matrix with zero diagonal.

    Yields: (sub_W, active_indices): submatrix (sparse) and active indices.

    See Algorithm 1 from Zhuang et al. http://arxiv.org/abs/1603.02844

    """
    num_elements = W.shape[0]
    U = set(range(num_elements))
    while U:
        cur_idx = choice(list(U))
        active_indices = [cur_idx]

        print ("cur", cur_idx, U)
        U.remove(cur_idx)
        possible_indices = np.nonzero(W[cur_idx, :].A.ravel() < 0)[0].tolist()
        print ("poss", possible_indices)
        shuffle(possible_indices)
        for p in possible_indices:
            if np.all(W[p, :].A.ravel()[active_indices] <= 0):
                active_indices.append(p)
                U.discard(p)
        active_indices = sorted(active_indices)
        print("act", active_indices)
        yield W[active_indices, :][:, active_indices].A, active_indices

def solve_iterative(W, iterations=10):

    """Find an approximate minimizer of (x.T * W * x) for x_i in {-1, 1}.

    Args:
        W: matrix of weights.
        iterations: number of iterations of finding submodular sub-matrices and
           solving them in random order.

    Returns:
        vector x of {-1, 1}.

    See Algorithm 1 from Zhuang et al. http://arxiv.org/abs/1603.02844

    """

    # ignore the diagonal, and symmetrize
    W = W.copy()
    W.setdiag(0)
    W = (W + W.T) / 2

    num_elements = W.shape[0]

    # initial guess of all 1s
    current_labels = 0 * (np.random.randint(2, size=num_elements) * 2 - 1)

    for _ in range(iterations):
        for sub_W, active_indices in generate_submodular(W):
            sub_W = sub_W.astype(np.float32)  # pymaxflow compatibility
            num_active = len(active_indices)

            print("solve", active_indices, "\n", sub_W)

            # We are solving for active_indices.
            # All others are held fixed.
            # These are the costs relative to the labels we are holding fixed.
            costs_wrt_fixed = W.dot(current_labels)[active_indices]
            print("costs", costs_wrt_fixed)
            costs_wrt_fixed = costs_wrt_fixed.astype(np.float32)  # pymaxflow compatibility

            # split into positive and negative pieces
            pos_costs_wrt_fixed = costs_wrt_fixed.copy()
            neg_costs_wrt_fixed = - costs_wrt_fixed
            pos_costs_wrt_fixed[pos_costs_wrt_fixed < 0] = 0
            neg_costs_wrt_fixed[neg_costs_wrt_fixed < 0] = 0

            # set up a graphcut problem for this sub_W.
            # source = 1, sink = -1 (later mapped to 0).
            # Costs should be mapped to cuts.
            #
            #     source == 1
            #        |
            #        |  cost of A == -1  (neg_costs_wrt_fixed)
            #        |
            #        A
            #        |
            #        | cost of A == 1    (pos_costs_wrt_fixed)
            #        |
            #      sink == -1

            g = pymaxflow.PyGraph(num_active, num_active ** 2)
            g.add_node(num_active)
            # add max capacity to ensure flow is possible.  (I'm not sure this is necessary.)
            g.add_tweights_vectorized(np.arange(num_active, dtype=np.int32),
                                      2 * neg_costs_wrt_fixed,  # source capacity
                                      2 * pos_costs_wrt_fixed)  # sink capacity

            # add edges between labels
            row, col = np.nonzero(sub_W)
            # multiply by -2 because:
            #   if labels i,j are the same, score += W[i, j]
            #   if labels i,j are different (i.e., 1 & -1), score -= W[i, j]
            #   relative cost of separating vs. not = -2 * W[i, j]
            vals = -2 * sub_W[row, col]
            assert np.all(vals >= 0)
            g.add_edge_vectorized(row.astype(np.int32), col.astype(np.int32), vals, vals)

            g.maxflow()
            out = (g.what_segment_vectorized() == pymaxflow.SOURCE) * 2 - 1
            print("res", out)

            # validate solution
            best = np.inf
            for possible in itertools.product([-1, 1], repeat=num_active):
                v = np.array(possible)
                e = sub_W.dot(v).dot(v) + costs_wrt_fixed.dot(v)
                if e < best:
                    best_v = v
                    best = e
            print("best", best_v, best)

            # set active labels to new configuration
            current_labels[active_indices] = out
            print ("NEW", best_v, active_indices, current_labels)
            print("")

    print ("C", current_labels, W.dot(current_labels).dot(current_labels))
    return (current_labels + 1) / 2


def find_next_bits(current_hashes, labels, triplet_loss, hashlen):
    """Find the next bit to append to each hash value to greedily minimze a triplet loss.


    Args:
        current_hashes: a list or array current hash values
            New hashes will be 2x these values + (0 or 1)
            NB: be sure representation has sufficient range.
        labels: list or array of true labels
        triplet_loss: function that can be called with arguments:
            (hash1, hash2, has3) where label1 == label2 != label3
            and returns the loss for that triple.
        hashlen: number of bits in current hashes

    Returns:
        new hashes approximately minimizing the triplet loss over
        all pairs.

    See section 3.1 of
    Fast Training of Triplet-based Deep Binary Embedding Networks
    Bohan Zhuang, Guosheng Lin, Chunhua Shen, Ian Reid
    http://arxiv.org/abs/1603.02844

    """

    old_loss = compute_loss(current_hashes, labels, triplet_loss, hashlen)
    print("old", old_loss)

    print("finding W")
    W = find_W(current_hashes, labels, triplet_loss, hashlen)

    print("solving for new bits")

    best_loss = np.inf
    better_count = 0
    for _ in range(10):
        new_bits = solve_iterative(W).astype(int).tolist()
        # choose new hashes
        # new_bits = [(idx % 2) for idx in range(len(current_hashes))]
        new_hashes = [h * 2 + o for h, o in zip(current_hashes, new_bits)]
        new_loss = compute_loss(new_hashes, labels, triplet_loss, hashlen + 1)
        print("   LOSS", new_loss, new_loss < best_loss, best_loss < old_loss)
        if new_loss < best_loss:
            best_loss = new_loss
            best_new_bits = new_bits
        if new_loss < old_loss:
            better_count += 1
        if better_count == 3:
            break

    # convert to 1/-1 vector
    bits_as_vec = np.matrix(best_new_bits) * 2 - 1
    print (W.todense())
    print("delta v. pred", best_loss - old_loss, bits_as_vec.dot(W.dot(bits_as_vec.T)), best_loss)

    for idx in range(len(current_hashes)):
        bits_as_vec[0, idx] = - bits_as_vec[0, idx]
        print("flip", idx, bits_as_vec.dot(W.dot(bits_as_vec.T)))
        bits_as_vec[0, idx] = - bits_as_vec[0, idx]
    best_new_hashes = [h * 2 + o for h, o in zip(current_hashes, best_new_bits)]
    return best_new_hashes

def hamming(h1, h2):
    """ hamming distance between h1 and h2 """
    return bin(h1 ^ h2).count('1')


def hinged_triplet_loss(h1, h2, h3, bitlen):
    # print(h1, h2, h3, "->", max(0, bitlen / 2.0 - (hamming(h1, h3) - hamming(h1, h2))))
    # h1 same label as h2, different label from h3
    bitdiff = (hamming(h1, h3) - hamming(h1, h2))
    return max(0, bitlen / 2.0 - bitdiff)

if __name__ == '__main__':
    labels = ([0] * 3) + ([1] * 3) + ([3] * 5) + ([4] * 5)
    hashes = [0] * 3 + [1] * 3 + [5] * 5 + list(range(5))
    l = 4
    for iter in range(64):
        print("\n\n\nITER")
        print(iter)
        hashes = find_next_bits(hashes, labels, hinged_triplet_loss, l)
        l = l + 1
        for h in hashes:
            print(bin(h)[2:].zfill(l))
