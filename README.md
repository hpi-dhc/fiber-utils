# fiber-utils

Utility methods for FIBER that do not belong into the core module.
Like for visualizations, cohort filters, or time series transformation.

* Free software: MIT license

## Installation

From source:

```
git clone https://gitlab.hpi.de/fiber/fiber-utils && \
cd fiber-utils && \
pip install -e .
```

## Categories of Utility Functions

* Cohort utilities
    * Remove duplicated codings & filter cohorts based on information density
        * `days_between_cohort_condition_occurrences`
        * `deduplicate_cohort`
        * `cohort_condition_occurrence_heatmap`
        * `cohort_condition_occurrence_filter`
    * Compare cohorts
        * `cohort_overlap`
    * Time series handling
        * `pivot_time_series`
* Condition utilities
    * Condition incidence and occurrence exploration and plotting
        * `most_common_condition_codes_for_cohort`
        * `number_condition_occurrences_per_patient`
        * `condition_occurrence_distribution`
        * `compare_condition_incidence_in_cohort`
        * `plot_condition_first_occurrence_on_day_hist`
        * `plot_condition_last_occurrence_on_day_hist`
    * Quantiles & days until quantiles
        * `condition_occurrence_quantiles_for_days`
        * `condition_occurrence_number_days_for_quantiles`
* Time series utilities (symbolic and sparse symbolic representation)
    * `sax_transform`
    * `ssr_transform`

The implementation of Sparse Symbolic Representation is based on https://github.com/fbagattini/sparse_symbolic_representation.
Usage granted through Creative Commons License in https://doi.org/10.1186/s12911-018-0717-4.
