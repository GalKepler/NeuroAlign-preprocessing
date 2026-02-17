import pytest
from neuroalign_preprocessing.naming_utils import generate_parquet_filename

def test_generate_parquet_filename_cat12_with_mask():
    entities = {
        'sub': 'CLMC10',
        'ses': '202407110849',
        'atlas': 'Schaefer2018N400n7Tian2020S3',
        'tissue': 'GM'
    }
    filename = generate_parquet_filename(
        entities=entities,
        modality='cat12',
        mask='gm',
        maskthr=50
    )
    expected = (
        "sub-CLMC10_ses-202407110849_atlas-Schaefer2018N400n7Tian2020S3_"
        "space-MNI152NLin2009cAsym_res-01_tissue-GM_mask-gm_maskthr-50_parc.parquet"
    )
    assert filename == expected

def test_generate_parquet_filename_cat12_no_mask():
    entities = {
        'sub': 'CLMC10',
        'ses': '202407110849',
        'atlas': 'Schaefer2018N400n7Tian2020S3',
        'tissue': 'WM'
    }
    filename = generate_parquet_filename(
        entities=entities,
        modality='cat12',
        mask=None,
        maskthr=None
    )
    expected = (
        "sub-CLMC10_ses-202407110849_atlas-Schaefer2018N400n7Tian2020S3_"
        "space-MNI152NLin2009cAsym_res-01_tissue-WM_parc.parquet"
    )
    assert filename == expected

def test_generate_parquet_filename_qsiparc_with_mask():
    entities = {
        'sub': 'CLMC10',
        'ses': '202407110849',
        'atlas': 'Schaefer2018N400n7Tian2020S3',
        'model': 'tensor',
        'param': 'md'
    }
    filename = generate_parquet_filename(
        entities=entities,
        modality='qsiparc',
        mask='gm',
        maskthr=50
    )
    expected = (
        "sub-CLMC10_ses-202407110849_atlas-Schaefer2018N400n7Tian2020S3_"
        "space-MNI152NLin2009cAsym_res-01_model-tensor_param-md_mask-gm_maskthr-50_parc.parquet"
    )
    assert filename == expected

def test_generate_parquet_filename_qsiparc_no_mask():
    entities = {
        'sub': 'CLMC10',
        'ses': '202407110849',
        'atlas': 'Schaefer2018N400n7Tian2020S3',
        'model': 'csd',
        'param': 'fa'
    }
    filename = generate_parquet_filename(
        entities=entities,
        modality='qsiparc',
        mask=None,
        maskthr=None
    )
    expected = (
        "sub-CLMC10_ses-202407110849_atlas-Schaefer2018N400n7Tian2020S3_"
        "space-MNI152NLin2009cAsym_res-01_model-csd_param-fa_parc.parquet"
    )
    assert filename == expected

def test_generate_parquet_filename_minimal_entities():
    entities = {
        'sub': 'sub01',
        'ses': 'ses01',
        'atlas': 'myatlas',
        'tissue': 'GM'
    }
    filename = generate_parquet_filename(
        entities=entities,
        modality='cat12'
    )
    expected = (
        "sub-sub01_ses-ses01_atlas-myatlas_"
        "space-MNI152NLin2009cAsym_res-01_tissue-GM_parc.parquet"
    )
    assert filename == expected

def test_generate_parquet_filename_no_sub_ses_in_entities():
    # This scenario is for the aggregated files, where sub/ses are not in filename
    entities = {
        'atlas': 'Schaefer2018N400n7Tian2020S3',
        'tissue': 'GM'
    }
    filename = generate_parquet_filename(
        entities=entities,
        modality='cat12',
        mask='gm',
        maskthr=50
    )
    expected = (
        "atlas-Schaefer2018N400n7Tian2020S3_"
        "space-MNI152NLin2009cAsym_res-01_tissue-GM_mask-gm_maskthr-50_parc.parquet"
    )
    assert filename == expected

def test_generate_parquet_filename_qsiparc_no_sub_ses_in_entities():
    entities = {
        'atlas': 'Schaefer2018N400n7Tian2020S3',
        'model': 'tensor',
        'param': 'md'
    }
    filename = generate_parquet_filename(
        entities=entities,
        modality='qsiparc',
        mask='gm',
        maskthr=50
    )
    expected = (
        "atlas-Schaefer2018N400n7Tian2020S3_"
        "space-MNI152NLin2009cAsym_res-01_model-tensor_param-md_mask-gm_maskthr-50_parc.parquet"
    )
    assert filename == expected
