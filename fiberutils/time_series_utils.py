"""
Implementation of Sparse Symbolic Representation based on
https://github.com/fbagattini/sparse_symbolic_representation
Usage granted through Creative Commons in
https://doi.org/10.1186/s12911-018-0717-4.
"""
from math import log

import editdistance as ed
import numpy as np
from fiber.config import OCCURRENCE_INDEX
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa
from saxpy.sax import ts_to_string
from saxpy.znorm import znorm


def _preprocess_ts(time_series_df):
    del time_series_df["time_delta_in_days"]
    time_series_df["value_representation"] = znorm(
        time_series_df.numeric_value
    )
    del time_series_df["numeric_value"]
    return time_series_df


def sax_transform(time_series_df, num_cuts):
    time_series_df = _preprocess_ts(time_series_df)
    cuts = cuts_for_asize(num_cuts)
    return (
        time_series_df.groupby(OCCURRENCE_INDEX)
        .agg(lambda x: ts_to_string(x.to_numpy(), cuts))
        .reset_index()
    )


def paa_transform(time_series_df, num_cuts, len_agg):
    time_series_df = _preprocess_ts(time_series_df)
    cuts = cuts_for_asize(num_cuts)
    return (
        time_series_df.groupby(OCCURRENCE_INDEX)
        .agg(lambda x: ts_to_string(paa(x.to_numpy(), len_agg), cuts))
        .reset_index()
    )


def _get_random_subsequence(sequences):
    """
    generate a random subsequence from a list of alphabetic sequences
    """
    selected_sequence = sequences[np.random.randint(len(sequences))]
    if len(selected_sequence) == 1:
        return selected_sequence
    random_offset = np.random.randint(len(selected_sequence))
    random_length = np.random.randint(1, len(selected_sequence))
    return selected_sequence[random_offset: (random_offset + random_length)]


def _sliding_ed(sequence, shapelet):
    """
    compute the minimum edit distance between a shapelet and
    (each subsequence of) a sequence
    """
    if len(shapelet) > len(sequence):
        return ed.eval(sequence, shapelet)
    return sorted(
        [
            ed.eval(sequence[i: i + len(shapelet)], shapelet)
            for i in range(len(sequence) - len(shapelet) + 1)
        ]
    )[0]


def _evaluate_candidate(
    candidate,
    edit_distances,
    labels,
    label_entropy,
    missing="lr",
    missing_data_labels=None,
):
    """
    evaluate a subsequence candidate
    """
    # sort labels wrt the edit distance
    sorted_distances, sorted_labels = zip(
        *[(e, l) for (e, l) in sorted(zip(edit_distances, labels))]
    )
    # all sequences have the same distance from the candidate: cannot split
    if len(set(sorted_distances)) == 1:
        return {
            "subseq": candidate,
            "ig": -1,
            "z": 0,
            "margin": 0,
        }

    # get all possible splits based on a threshold on the edit distance...
    all_splits = _get_all_splits(sorted_labels, sorted_distances)

    # ...and compute the corresponding ig and margin
    if missing == "lr":
        evaluations = [
            {
                "subseq": candidate,
                "ig": max(
                    label_entropy
                    - _get_split_entropy(
                        missing_data_labels + list(s[0]), s[1]
                    ),
                    label_entropy
                    - _get_split_entropy(
                        s[0], list(s[1]) + missing_data_labels
                    ),
                ),
                # which value will be assigned to missing data at
                # transformation time
                "z": 0
                if _get_split_entropy(missing_data_labels + list(s[0]), s[1])
                < _get_split_entropy(s[0], list(s[1]) + missing_data_labels)
                else max(edit_distances) + 1,
                "margin": sorted_distances[len(s[0])]
                - sorted_distances[len(s[0]) - 1],  # break ig ties
                "threshold": sorted_distances[len(s[0])],
                "index": i,  # to preserve the order in case of ig+margin ties
            }
            for i, s in enumerate(all_splits)
        ]

    else:
        evaluations = [
            {
                "subseq": candidate,
                "ig": label_entropy - _get_split_entropy(*s),
                "z": None,
                "margin": sorted_distances[len(s[0])]
                - sorted_distances[len(s[0]) - 1],
                "threshold": sorted_distances[len(s[0])],
                "index": i,
            }
            for i, s in enumerate(all_splits)
        ]

    # return the split yielding the maximum gain (margin is used to break ties)
    best_evaluation = sorted(
        evaluations, key=lambda e: (-e["ig"], -e["margin"], e["index"])
    )[0]
    print("best split:", best_evaluation)
    return best_evaluation


def _entropy(labels):
    """
    compute the entropy of a label set
    """
    n = len(labels)
    pos = sum(labels)
    pos_ratio = pos / n
    neg_ratio = (n - pos) / n
    return (
        -0.0
        if not pos_ratio or not neg_ratio
        else -(pos_ratio * log(pos_ratio, 2) + neg_ratio * log(neg_ratio, 2))
    )


def _get_all_splits(sorted_labels, sorted_distances):
    """
    compute all the possible ways of separating a labeled set based on a
    threshold on the edit distance
    """
    return [
        (sorted_labels[: split_index + 1], sorted_labels[split_index + 1:])
        for split_index in np.where(np.diff(sorted_distances))[0]
    ]


def _get_split_entropy(a, b):
    """
    compute the entropy of a split (two separated label sets)
    """
    tot_len = len(a) + len(b)
    return len(a) / tot_len * _entropy(a) + len(b) / tot_len * _entropy(b)


def ssr_transform(
    time_series_df, cohort, onset_df, num_cuts=3, missing='lr', n_candidates=20
):
    """
    Sparse symbolic representation of time series
    """

    target = list(
        set(onset_df.columns) - {"medical_record_number", "age_in_days"}
    )[0]
    df = cohort.merge_patient_data(
        sax_transform(time_series_df=time_series_df, num_cuts=3,), onset_df
    ).fillna("z")
    seqs, labels = df["value_representation"], df[target]

    # get non-empty seqs and their labels
    actual_seqs, actual_labels = zip(
        *[(_, labels[i]) for i, _ in enumerate(seqs) if _ != "z"]
    )

    # generate unique candidates; sort them to preserve order
    candidates = set(
        [_get_random_subsequence(actual_seqs) for _ in range(n_candidates)]
    )
    candidates = list(sorted(candidates))

    # evaluate candidates according to 'lr' or 'plain' method
    print("# Candidate evaluation")
    if missing == "lr":
        missing_data_labels = [
            l for i, l in enumerate(labels) if seqs[i] == "z"
        ]
        candidate_evals = [
            _evaluate_candidate(
                c,
                [_sliding_ed(s, c) for s in actual_seqs],
                actual_labels,
                _entropy(labels),
                missing_data_labels=missing_data_labels,
            )
            for c in candidates
        ]

    else:  # 'plain'
        candidate_evals = [
            _evaluate_candidate(
                c,
                [_sliding_ed(s, c) for s in seqs],
                labels,
                _entropy(labels),
                missing="plain",
            )
            for c in candidates
        ]

    # select candidate (shapelet) yielding maximum information gain (to break
    # ties, max margin and min length)
    shapelet = sorted(
        candidate_evals,
        key=lambda e: (
            -e["ig"],  # max ig
            -e["margin"],  # max margin
            len(e["subseq"]),
        ),
    )[
        0
    ]  # min length

    print("# Shapelet selection")
    print(
        "selected shapelet:{} ig:{:.3f} margin:{}".format(
            shapelet["subseq"], shapelet["ig"], shapelet["margin"]
        )
    )

    # transform sequence dataset based on the selected shapelet
    if missing == "lr":
        transformed_seqs = [
            _sliding_ed(s, shapelet["subseq"]) if s != "z" else shapelet["z"]
            for s in seqs
        ]
    else:
        transformed_seqs = [_sliding_ed(s, shapelet["subseq"]) for s in seqs]

    df['value_representation'] = transformed_seqs
    return df[[
        'medical_record_number',
        'age_in_days',
        'value_representation'
    ]]
