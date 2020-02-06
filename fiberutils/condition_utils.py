import math

import matplotlib.pyplot as plt
import pandas as pd
from fiber.plots.distributions import hist


def _fetch_data(
    condition,
    cohort,
    lower_limit=-math.inf,
    upper_limit=math.inf
):
    """
    Fetch cohort values for a condition with a time limit

    Args:
        condition: condition data to fetch
        cohort: cohort to fetch the data for
        lower_limit: days limit before the cohort condition
        upper_limit: days limit after the cohort condition
    """

    condition.data_columns = [
        condition.mrn_column,
        condition.age_column,
        condition.code_column,
        condition.description_column
    ]

    df = cohort.values_for(
        target=condition,
        relative_to=cohort.condition
    )

    return df[
        (df.time_delta_in_days > lower_limit)
        & (df.time_delta_in_days < upper_limit)
    ]


def _get_col_names(condition):
    """Helper method to return column names for conditions."""
    return (
        condition.mrn_column.name.lower(),
        condition.code_column.name.lower(),
        condition.description_column.name.lower()
    )


def _code_counts_per_window(
    df,
    condition,
    time_windows
):
    """
    Count code column values for a given DataFrame in occurrence format within
    different time windows.
    """
    mrn_column, code_column, description_column = _get_col_names(condition)

    results = {}
    for lower_limit, upper_limit in time_windows:
        label = f'{lower_limit} to {upper_limit} days'
        view = df[
            (df.time_delta_in_days >= lower_limit)
            & (df.time_delta_in_days <= upper_limit)
        ]
        result = view.groupby([
            code_column, description_column
        ]).count().reset_index().rename(
            columns={
                mrn_column: 'count',
                code_column: 'code',
                description_column: 'description'
            }
        ).sort_values(
            by='count',
            ascending=False,
        )[[
            'code',
            'description',
            'count'
        ]]

        results[label] = result

    return results


def _incidence_rates_per_window(
    df,
    cohort,
    time_windows=[[0, math.inf], [0, 365], [0, 30], [0, 7]]
):
    """
    Calculate the percentage of MRNs in a cohort that fulfil a condition
    denoted in an occurence df within different time windows.
    """
    results = {}
    for lower_limit, upper_limit in time_windows:
        label = f'{lower_limit} to {upper_limit} days'
        mrns = df[
            (df.time_delta_in_days >= lower_limit)
            & (df.time_delta_in_days <= upper_limit)
        ].medical_record_number.unique()

        results[label] = len(mrns) / len(cohort)
    return results


def _generate_distribution_count_figures(df, time_windows):
    time_deltas = df.time_delta_in_days.rename(
        '# occurrences on days'
    )
    results = {}
    for lower_limit, upper_limit in time_windows:
        label = f'{lower_limit} to {upper_limit} days'

        if abs(upper_limit) == math.inf or abs(lower_limit) == math.inf:
            bins = None
        else:
            bins = abs(upper_limit - lower_limit) or 1

        results[label] = hist(
            series=time_deltas[
                (time_deltas < upper_limit)
                & (time_deltas >= lower_limit)
            ],
            kde=False,
            bins=bins
        )

    return results


def _generate_distribution_incidence_figures(incidence_rates):
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_xlabel('incidence rate on interval of days relative to cohort condition') # noqa

    labels = []
    for c in incidence_rates.keys():
        labels.append(c.replace(' days', ''))

    plt.bar(
        x=labels,
        height=incidence_rates.values()
    )
    plt.setp(ax.get_xticklabels(), rotation=90)
    return fig


def condition_occurrence_distribution(
    cohort,
    condition,
    time_windows=[[-math.inf, math.inf]]
):
    """
    Describe the distribution of condition occurrences for a cohort over
    different time windows including incidence rates.
    """
    mrn_column, code_column, description_column = _get_col_names(condition)
    df = _fetch_data(condition, cohort, -math.inf, math.inf)

    incidence_rates = _incidence_rates_per_window(
        df,
        cohort,
        time_windows
    )

    return {
        'time_deltas': df.time_delta_in_days,
        'code_counts': {
            'results': _code_counts_per_window(df, condition, time_windows),
            'figures': _generate_distribution_count_figures(df, time_windows),
        },
        'incidence_rates': {
            'results': incidence_rates,
            'figure': _generate_distribution_incidence_figures(
                incidence_rates
            )
        }
    }


def condition_occurrence_quantiles_for_days(
    cohort,
    condition,
    lower_limit,
    upper_limit,
    observation_limits,
    return_raw=False
):
    if (
        not type(observation_limits) == list
    ) or (
        min(observation_limits) < lower_limit
    ) or (
        max(observation_limits) > upper_limit
    ):
        raise AttributeError(
            '''
            Provide a list of integer values between the
            lower and upper limit
            '''
        )

    df = _fetch_data(condition, cohort, lower_limit, upper_limit)
    time_deltas = df.time_delta_in_days

    quantiles = {}
    for x in observation_limits:
        quantiles[x] = len(time_deltas[time_deltas < x]) / len(time_deltas)

    result = {
        'quantiles': quantiles
    }

    if return_raw:
        result['time_deltas'] = time_deltas

    return result


def condition_occurrence_number_days_for_quantiles(
    cohort,
    condition,
    lower_limit,
    upper_limit,
    quantiles,
    return_raw=False
):
    if (
        not type(quantiles) == list
    ) or (
        min(quantiles) < 0
    ) or (
        max(quantiles) > 1
    ):
        raise AttributeError(
            '''
            Provide a list of float values between the
            lower and upper limit
            '''
        )

    df = _fetch_data(condition, cohort, lower_limit, upper_limit)
    time_deltas = df.time_delta_in_days

    no_days = {}
    for x in quantiles:
        no_days[x] = time_deltas.quantiles(x)

    result = {
        'no_days': no_days
    }

    if return_raw:
        result['time_deltas'] = time_deltas

    return result


def most_common_condition_codes_for_cohort(
    condition,
    cohort,
    lower_limit=-math.inf,
    upper_limit=math.inf
):
    """
    Calculate the number of patients associated with a condition as well as
    the incidence rate of the condition within the time window.

    Args:
        condition: condition to calculate the incidence for
        cohort: cohort of patients to calculate the condition incidence for
        lower_limit: days before the cohort condition to start the interval
        upper_limit: days after the cohort condition to end the interval
    """
    mrn_column, code_column, description_column = _get_col_names(condition)
    df = _fetch_data(condition, cohort, lower_limit, upper_limit)

    result = df[[
        mrn_column, code_column
    ]].groupby(
        code_column
    ).agg(
        lambda x: len(x[mrn_column].unique())
    ).rename(
        columns={mrn_column: 'count'}
    ).reset_index()

    sorted_result = df[[
        code_column,
        description_column
    ]].drop_duplicates().merge(
        result
    ).rename(
        columns={
            code_column: 'code',
            description_column: 'description'
        }
    ).sort_values(
        by='count',
        ascending=False
    )

    sorted_result['incidence_rate'] = sorted_result['count'] / len(cohort)
    return sorted_result.reset_index().drop('index', axis=1)


def compare_condition_incidence_in_cohort(
    condition,
    cohort,
    lower_limit=-math.inf,
    upper_limit=math.inf,
    should_calculate_increase=True,
    is_aggregated_condition=False
):
    """
    Compare the incidence of a condition for a cohort before and after
    the cohort condition with a given time interval.

    Args:
        condition: condition to find the incidence rates for
        cohort: patients to calculate the condition incidence for
        lower_limit: how many days before the cohort condition to start
        upper_limit: how many days after the cohort condition to end
        should_calculate_increase: should calculate the increase in incidence
        is_aggregated_condition: if using an aggregated condition and want the
            incidence rates calculated by sub-condition, e.g. by CPT-4 code
    """
    if is_aggregated_condition:
        occurrence_distribution = condition_occurrence_distribution(
            cohort=cohort,
            condition=condition,
            time_windows=[[lower_limit, -1], [0, upper_limit]]
        )

        figures = list(
            occurrence_distribution['code_counts']['figures'].values()
        ) + [occurrence_distribution['incidence_rates']['figure']]
        incidence_rates = list(
            occurrence_distribution['incidence_rates']['results'].values()
        )

        result = pd.Series({
            'count_before': incidence_rates[0] * len(cohort),
            'count_after': incidence_rates[1] * len(cohort),
            'incidence_rate_before': incidence_rates[0],
            'incidence_rate_after': incidence_rates[1],
        })

        if should_calculate_increase:
            result['increase'] = incidence_rates[1] / incidence_rates[0]

    else:
        before = most_common_condition_codes_for_cohort(
            condition=condition,
            cohort=cohort,
            lower_limit=lower_limit,
            upper_limit=-1
        )
        after = most_common_condition_codes_for_cohort(
            condition=condition,
            cohort=cohort,
            lower_limit=0,
            upper_limit=upper_limit
        )

        result = pd.merge(
            left=before,
            right=after,
            how='outer',
            suffixes=['_before', '_after'],
            on=['code', 'description']
        ).sort_values(
            by='count_after', ascending=False
        ).reset_index().drop(columns='index')

        if should_calculate_increase:
            result['increase'] = result['count_after'] / result['count_before']
            result = result.sort_values(
                by='increase', ascending=False
            ).reset_index().drop(columns='index')

        figures = None  # TODO generate graph to compare condition incidences

    return result, figures


def plot_condition_first_occurrence_on_day_hist(
    cohort,
    condition,
    lower_limit=-365,
    upper_limit=0
):
    df = cohort.values_for(
        target=condition,
        before=cohort.condition,
    )

    df = df[[
        'medical_record_number', 'age_in_days', 'time_delta_in_days'
    ]].drop_duplicates()
    df = df[
        (df.time_delta_in_days >= lower_limit)
        & (df.time_delta_in_days <= upper_limit)
    ]
    df = df.groupby([
        'medical_record_number', 'age_in_days'
    ]).min().drop_duplicates()
    return {
        'data': df.drop_duplicates(),
        'figure': df.time_delta_in_days.hist(
            bins=int(abs((upper_limit - lower_limit) / 7))
        )
    }


def plot_condition_last_occurrence_on_day_hist(
    cohort,
    condition,
    lower_limit=-365,
    upper_limit=0
):
    df = cohort.values_for(
        target=condition,
        before=cohort.condition,
    )

    df = df[[
        'medical_record_number', 'age_in_days', 'time_delta_in_days'
    ]].drop_duplicates()
    df = df[
        (df.time_delta_in_days >= lower_limit)
        & (df.time_delta_in_days <= upper_limit)
    ]
    df = df.groupby([
        'medical_record_number', 'age_in_days'
    ]).max().drop_duplicates()

    return {
        'data': df,
        'figure': df.time_delta_in_days.hist(
            bins=int(abs((upper_limit - lower_limit) / 7))
        )
    }


def number_condition_occurrences_per_patient(
    cohort,
    condition,
    time_interval=[-365, 0],
    threshold=20
):
    df = cohort.values_for(
        target=condition,
        before=cohort.condition,
    )

    result = cohort.aggregate_values_in(
        time_windows=[time_interval],
        df=df,
        aggregation_functions={'time_delta_in_days': 'count'},
        name='code_occurrences'
    )

    count_column = result.columns[
        result.columns.str.contains('code_occurrences')
    ][0]
    result = result.rename(columns={
        count_column: 'code_occurrences'
    })
    result.loc[result.code_occurrences.isnull(), 'code_occurrences'] = 0

    return {
        'raw': result.drop_duplicates(),
        'figure': result[
            result.code_occurrences < threshold
        ].code_occurrences.hist(bins=threshold)
    }
