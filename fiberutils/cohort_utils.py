from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import venn
from fiber import Cohort
from fiber.condition import MRNs
from fiber.config import OCCURRENCE_INDEX, VERBOSE
from fiber.dataframe import (
    column_threshold_clip,
    create_id_column,
    merge_to_base,
    time_window_clip
)
from fiber.plots.distributions import bars

from fiberutils.time_series_utils import ssr_transform


def cohort_overlap(cohorts: dict):
    """
    Helper method to intersect MRNs from different cohorts
    and generate a venn diagram based on that.

    Args:
        cohorts: dict of label and Cohort object
    Returns:
        dict with overlapping MRNs and figure
    """
    cohort_list = list(cohorts.values())
    overlapping_mrns = cohort_list[0].mrns.copy()
    for cohort in cohort_list:
        overlapping_mrns &= cohort.mrns

    labels = venn.get_labels(
        [c.mrns for c in cohorts.values()]
    )
    figure, ax = venn.venn3(labels, names=cohorts.keys())

    return {
        'count': len(overlapping_mrns),
        'mrns': overlapping_mrns,
        'figure': figure
    }


def days_between_cohort_condition_occurrences(
    cohort: Cohort,
    limit: Optional[int] = None,
    return_raw: Optional[bool] = False
):
    """
    Count the number of days between occurrences of the cohort condition.
    This helps to find duplicated codings.
    Returns raw data as well as aggregated results and a figure.

    Args:
        cohort: FIBER cohort to calculate the differences for
        limit: upper limit of days between two occurrences for focus
        return_raw: whether to return the raw df and series
    """
    occurrences = cohort.occurrences
    df = occurrences.merge(
        occurrences.copy(),
        how='left',
        on='medical_record_number'
    )
    df['difference_in_days'] = df.age_in_days_x - df.age_in_days_y
    # all results are duplicated because of merge, symmetrically distributed
    df = df[df.difference_in_days >= 0]
    if limit:
        df = df[df.difference_in_days < limit]

    series = df.difference_in_days.rename('difference in days')
    result = {
        'counted': series.value_counts(),
        'figure': bars(series)
    }

    if return_raw:
        result = {
            **result,
            **{'df': df, 'series': series}
        }

    return result


def deduplicate_cohort(
    cohort,
    limit
):
    """
    Deduplicates cohort condition occurrences to have more accurate
    onsets of the cohort condition.
    Keeps only the first occurrence of the condition within the days limit.
    """
    duplicate_information = cohort.has_precondition(
        name='foo',
        condition=cohort.condition,
        time_windows=[[-1 * limit, -1]]
    )

    result_column = None
    for c in duplicate_information.columns:
        if c not in OCCURRENCE_INDEX:
            result_column = c
            break

    df = duplicate_information[
        ~duplicate_information[result_column]
    ][OCCURRENCE_INDEX]

    return {
        'cohort': Cohort(MRNs(df)),
        'df': df
    }


def _column_fill_percentage(df, column, lower_limit):
    return len(df[df[column] >= lower_limit]) / len(df)


def cohort_condition_occurrence_heatmap(
    cohort,
    condition,
    time_windows,
    max_condition_occurrences,
    should_annotate_figure,
    heatmap_kwargs={}
):
    """
    Heatmap of the fraction of patients in the cohort that have X condition
    occurrences (x-axis) in the specified time windows (y-axis).
    The condition can be any FIBER condition.
    """
    data = cohort.aggregate_values_in(
        time_windows=time_windows,
        df=cohort.values_for(
            condition,
            relative_to=cohort.condition
        ),
        aggregation_functions={
            'time_delta_in_days': lambda x: len(x.unique())
        }
    )

    results = {}
    for c in data.columns:
        if c not in OCCURRENCE_INDEX:
            fills = {}
            for limit in range(1, max_condition_occurrences + 1):
                fills[limit] = _column_fill_percentage(data, c, limit)
            results[c] = fills
    df = pd.DataFrame(results).T
    df.index = df.index.str.replace(
        'time_delta_in_days_interval_', ''
    ).str.replace(
        '_', ' '
    ).str.replace(
        'from ', ''
    ).str.replace(
        ' days', 'd'
    )
    df.index.name = 'time interval relative to cohort condition'

    plt.figure(figsize=(16, 9))
    plt.title(
        f'Fraction of patients with at least x number of {condition.__class__.__name__}'  # noqa
    )
    figure = sns.heatmap(
        data=df,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        linewidths=2,
        annot=bool(should_annotate_figure),
        **heatmap_kwargs
    )

    return {
        'data': df,
        'figure': figure
    }


def cohort_condition_occurrence_filter(
    cohort,
    condition,
    time_interval,
    encounter_lower_limit
):
    """
    Filter cohorts based on the number of encounters in a specific time
    interval, e.g. to remove patients that will not have dense data.
    """
    data = cohort.aggregate_values_in(
        time_windows=[time_interval],
        df=cohort.values_for(
            condition,
            relative_to=cohort.condition
        ),
        aggregation_functions={
            'time_delta_in_days': lambda x: len(x.unique())
        }
    )

    c = list(set(data.columns) - set(OCCURRENCE_INDEX))[0]
    data.rename(columns={c: 'no_encounters'}, inplace=True)

    return data[data.no_encounters >= encounter_lower_limit][
        OCCURRENCE_INDEX
    ].reset_index().drop(columns='index')


def get_time_series(cohort, condition, window, threshold=None):
    df = cohort.time_series_for(
        condition,
        before=cohort.condition,
        aggregate_value_per_day_func="mean"
    )
    df = time_window_clip(df=df, window=window)

    df.set_index(OCCURRENCE_INDEX, inplace=True)
    create_id_column(condition, df)
    if threshold:
        df = threshold_clip_time_series(
            df=df,
            cohort=cohort,
            threshold=threshold
        )

    return df


def threshold_clip_time_series(df, cohort, threshold):
        binarized_df = df.pivot_table(
            index=OCCURRENCE_INDEX,
            columns=['description'],
            aggfunc={'time_delta_in_days':'any'}
        )

        binarized_df.columns = binarized_df.columns.droplevel()
        binary_df_with_cohort = merge_to_base(
            cohort.occurrences,
            [binarized_df]
        ).set_index(OCCURRENCE_INDEX)

        cols_selected = column_threshold_clip(
            df=binary_df_with_cohort,
            threshold=threshold
        ).columns

        return df[df.description.isin(cols_selected)]


def pivot_time_series(
    cohort,
    onset_df,
    df
):
    """
    Fetch and pivot time series with sparse symbolic representation
    """
    if not df.empty:
        results = []
        for x in df.description.unique():
            if VERBOSE:
                print(f'############# {x} #############')
            current_df = df[df.description == x]

            transformed_df = ssr_transform(current_df, cohort, onset_df)
            transformed_df.rename(
                columns={'value_representation': x},
                inplace=True
            )
            results.append(transformed_df)
    else:
        results = [df]

    return merge_to_base(cohort.occurrences, results)
