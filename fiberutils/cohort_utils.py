from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import venn
from fiber import Cohort
from fiber.condition import MRNs
from fiber.config import OCCURRENCE_INDEX
from fiber.plots.distributions import bars


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
    ][[
        'medical_record_number',
        'age_in_days'
    ]]

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
        if c not in ['medical_record_number', 'age_in_days']:
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
        f'Percentage of patients with at least x number of {condition.__class__.__name__}'  # noqa
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

    for c in data.columns:
        if c not in ['medical_record_number', 'age_in_days']:
            data.rename(columns={c: 'no_encounters'}, inplace=True)

    return data[data.no_encounters >= encounter_lower_limit][[
        'medical_record_number',
        'age_in_days'
    ]].reset_index().drop(columns='index')
