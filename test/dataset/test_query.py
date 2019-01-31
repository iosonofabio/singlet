#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
import pytest


@pytest.fixture(scope="module")
def ds():
    from singlet.dataset import Dataset
    return Dataset(
            samplesheet='example_sheet_tsv',
            counts_table='example_table_tsv',
            featuresheet='example_sheet_tsv')


def test_average_samples(ds):
    print('Average samples')
    ds_avg = ds.average(axis='samples', column='experiment')
    assert(tuple(ds_avg.samplenames) == ('exp1', 'test_pipeline'))
    print('Done!')


def test_average_features(ds):
    print('Average features')
    ds_avg = ds.average(axis='features', column='annotation')
    assert(tuple(ds_avg.featurenames) == ('gene', 'other', 'spikein'))
    print('Done!')


def test_query_samples_meta(ds):
    print('Query samples by metadata')
    ds_tmp = ds.query_samples_by_metadata(
            'experiment == "test_pipeline"',
            inplace=False)
    assert(tuple(ds_tmp.samplenames) == ('test_pipeline',))
    print('Done!')


def test_query_sample_counts_onegene(ds):
    print('Query sample by counts in one gene')
    ds_tmp = ds.query_samples_by_counts('KRIT1 > 100', inplace=False)
    assert(tuple(ds_tmp.samplenames) == ('third_sample',))
    print('Done!')


def test_query_sample_total_counts(ds):
    print('Query sample by total counts')
    ds_tmp = ds.query_samples_by_counts('total < 3000000', inplace=False)
    assert(tuple(ds_tmp.samplenames) == ('second_sample',))
    print('Done!')


def test_query_mapped_counts(ds):
    print('Query sample by mapped counts')
    ds_tmp = ds.query_samples_by_counts('mapped < 1000000', inplace=False)
    assert(tuple(ds_tmp.samplenames) == ('second_sample',))
    print('Done!')


def test_query_features_counts(ds):
    print('Query features by counts')
    ds_tmp = ds.query_features_by_counts(
            'first_sample > 1000000',
            inplace=False)
    assert(tuple(ds_tmp.featurenames) == ('__alignment_not_unique',))
    print('Done!')
