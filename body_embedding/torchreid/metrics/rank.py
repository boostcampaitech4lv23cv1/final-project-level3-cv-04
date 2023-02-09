from __future__ import division, print_function, absolute_import
import numpy as np
import warnings
from collections import defaultdict

try:
    from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy
    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )


def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed num_repeats times.
    """
    num_repeats = 10
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc = 0.
        for repeat_idx in range(num_repeats):
            mask = np.zeros(len(raw_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_raw_cmc = raw_cmc[mask]
            _cmc = masked_raw_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)

        cmc /= num_repeats
        all_cmc.append(cmc)
        # compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )
    # print("### distmat")    
    # print(distmat) # shape(num_q, num_g) -> 8 x 60
    # '''
    # [[
    #     2.34557390e-01 2.21037984e-01 6.27429485e-02 2.33143568e-03
    #     2.17982709e-01 9.31802392e-02 2.96052158e-01 2.39819288e-01
    #     1.93972647e-01 2.29454041e-03 5.25697827e-01 2.21383035e-01
    #     8.69208455e-01 2.48094320e-01 1.62140310e-01 1.57529116e-03
    #     8.51219475e-01 8.72404516e-01 2.12396264e-01 8.49276125e-01
    #     1.62029266e-03 5.76301813e-01 5.22943497e-01 1.40564799e-01
    #     1.93692327e-01 1.88418210e-01 1.99521899e-01 2.54164577e-01
    #     2.05233932e-01 1.81343615e-01 1.67737246e-01 2.98151374e-01
    #     2.26159096e-02 2.32255459e-03 1.23023689e-01 1.21469557e-01
    #     8.92245173e-02 2.33290076e-01 2.30860054e-01 6.46440566e-01
    #     6.12040758e-02 8.00786018e-02 2.72096813e-01 2.28286028e-01
    #     2.16109753e-01 5.72629333e-01 4.96271849e-01 2.20873177e-01
    #     1.88386321e-01 9.18087959e-02 7.71265209e-01 2.59809077e-01
    #     1.80944443e-01 6.18636012e-01 2.13135242e-01 2.25234568e-01
    #     1.67106032e-01 2.74757683e-01 7.71737099e-02 2.54149139e-01]
    #     [4.29129243e-01 2.11108148e-01 3.54597747e-01 3.37885439e-01
    #     2.21564472e-01 3.48404169e-01 1.05763018e-01 4.23020005e-01
    #     3.40272248e-01 3.29099774e-01 9.40562487e-02 2.41168439e-01
    #     2.92314053e-01 4.37975347e-01 2.21310794e-01 3.29387903e-01
    #     2.86769390e-01 2.93493152e-01 1.98685348e-01 2.81641006e-01
    #     3.47185016e-01 6.56154156e-02 4.23233509e-02 4.34965312e-01
    #     2.64524758e-01 3.08518350e-01 1.88947082e-01 3.47862244e-02
    #     3.59899342e-01 3.20383787e-01 3.08235705e-01 6.41395628e-01
    #     3.89848471e-01 3.52453649e-01 1.12779021e-01 4.29836214e-01
    #     3.05429935e-01 4.33971941e-01 4.28784072e-01 2.56566644e-01
    #     3.74403596e-01 3.65051150e-01 3.01807702e-01 2.40150511e-01
    #     3.52480173e-01 1.17500424e-01 8.52446556e-02 2.41114199e-01
    #     3.42064619e-01 4.61792648e-01 2.35839844e-01 5.47249436e-01
    #     2.37500727e-01 1.37259841e-01 3.92685950e-01 3.00813019e-01
    #     1.83424354e-01 4.58227992e-02 3.40003192e-01 1.47173762e-01]
    #     [1.30923033e-01 1.40507638e-01 1.10164404e-01 1.35234714e-01
    #     1.40368998e-01 1.42680228e-01 3.04062247e-01 1.31529450e-01
    #     7.03417063e-02 1.26668274e-01 5.16089797e-01 1.43235445e-01
    #     9.26491201e-01 1.40062690e-01 1.49996400e-01 1.28809929e-01
    #     9.05399442e-01 9.27300692e-01 1.16569757e-01 9.07526195e-01
    #     1.32531464e-01 5.88354588e-01 5.10340631e-01 5.94334602e-02
    #     8.17927122e-02 1.36983335e-01 1.26918495e-01 2.23640621e-01
    #     7.71353245e-02 6.34117723e-02 1.09831035e-01 1.69006884e-01
    #     1.10356450e-01 1.35341883e-01 1.13430798e-01 1.11796856e-01
    #     7.91649818e-02 1.27406955e-01 1.32091463e-01 6.92290306e-01
    #     9.44283009e-02 7.98560977e-02 1.53816879e-01 1.53967559e-01
    #     1.27135396e-01 5.84830821e-01 4.90043283e-01 1.46782100e-01
    #     1.18470252e-01 1.07680082e-01 7.94266701e-01 1.51369810e-01
    #     9.47982073e-02 6.29870117e-01 8.05099607e-02 1.37412071e-01
    #     9.04667974e-02 2.55159438e-01 1.18424237e-01 2.31350720e-01]
    #     [2.86654353e-01 5.21165133e-03 2.73188353e-01 2.24854231e-01
    #     2.41029263e-03 2.90370882e-01 2.94583678e-01 2.81105220e-01
    #     1.85110211e-01 2.12755561e-01 4.77702618e-01 1.98018551e-03
    #     8.81639004e-01 3.02968800e-01 2.11475492e-01 2.10611284e-01
    #     8.69364500e-01 8.83070171e-01 3.41360569e-02 8.68816674e-01
    #     2.20741570e-01 4.41464186e-01 3.80068064e-01 2.14295030e-01
    #     4.55406308e-02 2.98623502e-01 4.07600403e-02 1.96502924e-01
    #     1.83956802e-01 1.32323921e-01 4.42549586e-02 3.70962918e-01
    #     2.31066048e-01 2.32615888e-01 1.46527827e-01 2.30914891e-01
    #     1.71347141e-01 2.83294380e-01 2.87143588e-01 6.82877660e-01
    #     1.99773729e-01 2.14165688e-01 3.99065018e-02 3.08018923e-03
    #     5.55236936e-02 5.28988242e-01 4.50663388e-01 2.65771151e-03
    #     4.39913869e-02 2.87609994e-01 7.85197675e-01 3.42159688e-01
    #     3.88762355e-02 5.83563626e-01 1.87039793e-01 3.04365158e-02
    #     6.10688329e-02 1.66158676e-01 2.57885098e-01 3.14689398e-01]
    #     [4.64431107e-01 2.24211454e-01 3.84161949e-01 3.64970386e-01
    #     2.35856533e-01 3.73568892e-01 1.04334712e-01 4.56781566e-01
    #     3.76783431e-01 3.54505897e-01 9.81868505e-02 2.54220426e-01
    #     2.87605286e-01 4.74665165e-01 2.45356560e-01 3.55661154e-01
    #     2.82802582e-01 2.88199544e-01 2.13505149e-01 2.76956916e-01
    #     3.73724103e-01 4.92984056e-02 2.72167921e-02 4.76556838e-01
    #     2.87709534e-01 3.36042881e-01 2.02930510e-01 4.47306037e-02
    #     3.94752562e-01 3.51101756e-01 3.29851210e-01 6.85912669e-01
    #     4.15947378e-01 3.80620122e-01 1.29656494e-01 4.55667138e-01
    #     3.33814383e-01 4.69870329e-01 4.64312434e-01 2.54323840e-01
    #     3.98764491e-01 3.97554755e-01 3.11602652e-01 2.52345145e-01
    #     3.71508241e-01 1.17486835e-01 8.95615816e-02 2.54345119e-01
    #     3.59355867e-01 4.97449756e-01 2.34449863e-01 5.88867664e-01
    #     2.53554881e-01 1.35898709e-01 4.27685440e-01 3.12588573e-01
    #     2.05171466e-01 5.07646203e-02 3.68177235e-01 1.58367991e-01]
    #     [2.07602978e-03 2.75396585e-01 1.80129290e-01 2.43942916e-01
    #     2.76598394e-01 1.98606908e-01 3.18720460e-01 1.51395798e-05
    #     9.10553336e-02 2.34408557e-01 5.33505023e-01 2.80798972e-01
    #     8.42530012e-01 2.30681896e-03 1.42229676e-01 2.35007942e-01
    #     8.20811510e-01 8.42850447e-01 2.98444748e-01 8.30065012e-01
    #     2.39285946e-01 6.59928083e-01 5.86863995e-01 1.26098216e-01
    #     2.73073137e-01 9.28621292e-02 2.58295000e-01 3.17809820e-01
    #     7.93268681e-02 9.48611498e-02 2.55214632e-01 9.17918682e-02
    #     2.17665195e-01 2.44952381e-01 2.08224893e-01 1.43256366e-01
    #     1.43073618e-01 2.75146961e-03 2.75266171e-03 6.33792162e-01
    #     2.38316000e-01 1.34045422e-01 3.78154278e-01 2.92135060e-01
    #     3.55167270e-01 5.88440597e-01 5.18136978e-01 2.83591688e-01
    #     2.86003828e-01 2.26263046e-01 7.41738915e-01 2.70407200e-02
    #     1.96635902e-01 6.37485385e-01 9.53158140e-02 3.28780174e-01
    #     2.19443202e-01 3.80873501e-01 1.82287335e-01 2.51142621e-01]
    #     [1.90808356e-01 1.63859189e-01 4.78508472e-02 4.60597277e-02
    #     1.63504958e-01 8.01336765e-02 2.23758221e-01 1.91751003e-01
    #     1.24978423e-01 4.19710875e-02 4.24934149e-01 1.67947114e-01
    #     7.74530172e-01 2.02488840e-01 1.14777565e-01 4.36694026e-02
    #     7.56585240e-01 7.76558638e-01 1.42204762e-01 7.57014751e-01
    #     4.63402867e-02 4.94699776e-01 4.34990287e-01 1.05286181e-01
    #     1.31486177e-01 1.45902276e-01 1.30178392e-01 1.74674571e-01
    #     1.41625643e-01 1.16840243e-01 1.27622962e-01 2.66407013e-01
    #     4.80053425e-02 4.64882851e-02 7.14738369e-02 1.02857709e-01
    #     7.10034966e-02 1.86696053e-01 1.88927531e-01 5.49517334e-01
    #     5.47924042e-02 5.28454185e-02 2.02235878e-01 1.76058650e-01
    #     1.63888097e-01 4.75322485e-01 3.98491502e-01 1.69172406e-01
    #     1.45314515e-01 7.52227306e-02 6.64104760e-01 2.29102850e-01
    #     1.19315028e-01 5.16307414e-01 1.48063838e-01 1.73818409e-01
    #     9.88478661e-02 2.17073023e-01 6.34029508e-02 1.71724319e-01]
    #     [3.03978205e-01 1.65041089e-02 2.86905229e-01 2.43170083e-01
    #     1.61040425e-02 3.14347208e-01 3.22954297e-01 2.98704267e-01
    #     1.93685234e-01 2.29704380e-01 5.13484359e-01 1.63512230e-02
    #     9.33585882e-01 3.21645677e-01 2.38462329e-01 2.27681994e-01
    #     9.19496655e-01 9.35114801e-01 3.86512876e-02 9.18888211e-01
    #     2.37976253e-01 4.68965888e-01 4.04947698e-01 2.16064453e-01
    #     4.43525314e-02 3.10252070e-01 5.39053679e-02 2.15587139e-01
    #     1.92730069e-01 1.41752839e-01 5.52209616e-02 3.76572013e-01
    #     2.39790678e-01 2.50913620e-01 1.58218503e-01 2.46022284e-01
    #     1.84182644e-01 3.01741600e-01 3.05920959e-01 7.31559157e-01
    #     2.06248939e-01 2.28038609e-01 3.11152935e-02 1.59563422e-02
    #     5.13153076e-02 5.70684254e-01 4.86153007e-01 1.57567859e-02
    #     4.76132035e-02 2.95646608e-01 8.30054700e-01 3.54141593e-01
    #     4.68665957e-02 6.26401067e-01 1.96868718e-01 2.56472230e-02
    #     6.10469580e-02 1.87066495e-01 2.78121889e-01 3.41657043e-01]]
    # '''
    # print("### q_pids")
    # print(q_pids) # [2 0 1 3 0 1 2 3] -> 8
    # print("### g_pids")
    # print(g_pids) 
    # '''
    # [1 3 2 2 3 2 2 1 1 2
    #  0 3 0 1 2 2 0 0 3 0
    #  2 0 0 1 3 1 3 0 1 1
    #  3 1 2 2 1 2 1 1 1 0
    #  2 2 3 3 3 0 0 3 3 2
    #  0 1 3 0 1 3 3 0 2 0]
    # '''-> 60
    # print("### g_camids")
    # print(g_camids) # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] -> 60
    # print("### max_rank")
    # print(max_rank) # 50

    indices = np.argsort(distmat, axis=1)
    # print("### indices")
    # print(indices)
    '''
    distmat value가 작은 순으로 정렬했을 때의 idx들, 0번째 query와 가장 닮은 gallery idx는 15라는 뜻
    indices = 
    [[15 20  9 33  3 32 36 41 58 35  2  5 49 40 34  1 29 26 52 31  4  0 38  7
    51 37 28 13 54 11 48 43 47  6  8 23 56 14 25 30 18 57 24 55 46 21 27 44
    10 45 42 39 22 53 16 12 19 17 50 59]
    [27 21 22 57 18 39  6 34 56  1 10 45 46 26 30  4 53 24 32 40 59 50 11 47
    48 43 14 35 23 52 44 49 29 28 54 55 31 41 42 16  8 25  2 58 12 19 17 36
    5 20 15  9  3 33 51 13  7 37 38  0]
    [34 56 29 23 40 54 57 28 31  8  1 24 32 26 35 18  6  4 49 48 27 30 52 41
    11 43 44 47 36 21 58  2 25  5 14 39 55 20 22 42 15 51 33  3  7  9 13 37
    38  0 46 10 45 53 59 50 16 12 19 17]
    [ 4 43 11 47  1 26 48 52 55 30 18 24 56 42 44 34 40 32 57 35 21 27  6 15
    20 29 36  9 41 31 49 33  3 54 23 28  8  2 58 22 14 39  5 51 25 46  7 13
    38  0 37 10 45 53 59 16 50 12 19 17]
    [27 21 57 18 22 56 34 24  6  1 39 26  4 40 30 48 44 11 43 47 32 52 42 10
    55 46 45 35 23 49 14 29 53 54 41 28 31 59  8 58  2 50 36  5 25 20 15  3
    9 33 16 12 17 19 51 13  7 37 38  0]
    [ 7 13  0 37 38 51 29 31 28 54 36  8 41  5 25 23 58 35 32  2  3 33 20 14
    9 34 49 15  6 40 46  1 10 26 56 45  4 52 57 39 53 48 11 43 16 47 30 12
    27 19 17 22 21 18 24 50 59 55 44 42]
    [40 34 32 49 56  1 26 41  4 35 48  6 57 18 52 29 24 11 36  2 43 20 47 58
    27 54 15 23 28 31  3 33 21  5  8  9 30 44 14 55 25 39 22 42 46 51 10  7
    13  0 38 37 45 53 59 50 16 12 17 19]
    [ 4 11 43 47  1 26 52 55 48 30 56 24 18 42 44 34 32 40 57 35 29  6 21 27
    31 15 54 36 23 20 28 41 49  9  8 33  3  2 58 14 39 22  5 25 51  7 13 38
    0 37 46 10 45 53 59 16 50 12 19 17]]
    '''
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # print("### matches")
    # print(matches)
    '''
    [
        [1 1 1 1 1 1 1 1 1 1
        1 1 0 0 1 0 0 0 0 0 
        0 0 1 0 0 0 0 1 0 0
        0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0]
        
        [1 1 1 1 1 1 1 1 1 1
        0 1 1 1 1 1 0 0 0 0
        0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0]
    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    [1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    [1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
    '''

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q): # 쿼리 수만큼 돌면서
        # get query pid and camid
        q_pid = q_pids[q_idx] # q_pid = 2 -> ninging
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        # pid, camid 가 같은 것들은 gallery에서 지우는 이유 : 다른 카메라에서 다시 나타나는지 확인하기 위해서
        # 우리는 하나의 카메라에서 찍기 떄문에 이 과정을 거칠 필요가 없다.
        order = indices[q_idx] # order = [15 20  9 33  3 32 36 41 58 35  2  5 49 40 34  1 29 26 52 31  4  0 38 51 37 28 13 54 11 48 43 47  6  8 23 56 14 25 30 18 57 24 55 46 21 27 44 10 45 42 39 22 53 16 12 19 17 50 59]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        # print(f"### q_idx : {q_idx} ### keep")
        # print(keep)

        # compute cmc curve
        # raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        raw_cmc = matches[q_idx]
        # print(f"### q_idx : {q_idx} ### raw_cmc")
        # print(raw_cmc)
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            print(" ⚠️ this condition is true when query identity does not appear in gallery")
            continue

        cmc = raw_cmc.cumsum()
        # print(f"### q_idx : {q_idx} ### cmc")
        # print(cmc)
        
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def evaluate_py(
    distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03
):
    if use_metric_cuhk03:
        return eval_cuhk03(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank
        )
    else:
        return eval_market1501(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank
        )


def evaluate_rank(
    distmat,
    q_pids,
    g_pids,
    q_camids,
    g_camids,
    max_rank=50,
    use_metric_cuhk03=False,
    use_cython=True
):
    """Evaluates CMC rank.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    if use_cython and IS_CYTHON_AVAI:
        return evaluate_cy(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03
        )
    else:
        return evaluate_py(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank,
            use_metric_cuhk03
        )
