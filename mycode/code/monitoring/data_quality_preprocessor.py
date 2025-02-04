# | filename: data_quality_preprocessor.py
# | code-line-numbers: true

import json


def preprocess_handler(inference_record, logger):
    input_data = inference_record.endpoint_input.data
    return {str(i).zfill(2): d for i, d in enumerate(input_data.split(","))}
