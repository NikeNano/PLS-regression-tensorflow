import logging

import apache_beam as beam
import numpy as np 

from .transpose import Transpose

from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to


def test_main():
    x = np.arange(4).reshape((2,2))
    logging.info(x)
    logging.info(np.transpose(x))


def test_main_two():
    pipeline_options=None
    with beam.Pipeline(options=pipeline_options) as p:
        lines = (p| beam.Create([
            {"row_key":0,"values":[0,0,0,0]}
            ,{"row_key":1,"values":[1,1,1,1]}
            ,{"row_key":2,"values":[2,2,2,2]}
            ,{"row_key":3,"values":[3,3,3,3]}
            ,{"row_key":3,"values":[4,4,4,4]}
            ]) | Transpose())


def test_main_three():
    pipeline_options=None
    with beam.Pipeline(options=pipeline_options) as p:
        lines = (p| beam.Create([
            {"row_key":0,"values":[0,1]}
            ,{"row_key":1,"values":[2,3]}
            ]) | Transpose())