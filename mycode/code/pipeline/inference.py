#| filename: inference.py
#| code-line-numbers: true

import os
import json
import requests
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def handler(data, context, directory=Path("/opt/ml/model")):
    """
    This is the entrypoint that will be called by SageMaker
    when the endpoint receives a request.
    """
    print("Handling endpoint request")

    processed_input = _process_input(data, context, directory)
    output = _predict(processed_input, context, directory) if processed_input else None
    return _process_output(output, context, directory)


def _process_input(data, context, directory):
    print("Processing input data...")

    if context is None:
        # The context will be None when we are testing the code
        # directly from a notebook. In that case, we can use the
        # data directly.
        endpoint_input = data
    elif context.request_content_type in (
        "application/json",
        "application/octet-stream",
    ):
        # When the endpoint is running, we will receive a context
        # object. We need to parse the input and turn it into
        # JSON in that case.
        endpoint_input = data.read().decode("utf-8")
    else:
        raise ValueError(
            f"Unsupported content type: {context.request_content_type or 'unknown'}"
        )

    # Let's now transform the input data using the features pipeline.
    try:
        endpoint_input = json.loads(endpoint_input)
        df = pd.json_normalize(endpoint_input)
        features_pipeline = joblib.load(directory / "features.joblib")
        result = features_pipeline.transform(df)
    except Exception as e:
        print(f"There was an error processing the input data. {e}")
        return None

    return result[0].tolist()


def _predict(instance, context, directory):
    print("Sending input data to model to make a prediction...")

    if context is None:
        # The context will be None when we are testing the code
        # directly from a notebook. In that case, we want to load the
        # model we trained and make a prediction using it.
        import keras

        model = keras.models.load_model(Path(directory) / "001")
        predictions = model.predict(np.array([instance]))
        result = {"predictions": predictions.tolist()}
    else:
        # When the endpoint is running, we will receive a context
        # object. In that case we need to send the instance to the
        # model to get a prediction back.
        model_input = json.dumps({"instances": [instance]})
        response = requests.post(context.rest_uri, data=model_input)

        if response.status_code != 200:
            raise ValueError(response.content.decode("utf-8"))

        result = json.loads(response.content)

    print(f"Response: {result}")
    return result


def _process_output(output, context, directory):
    print("Processing prediction received from the model...")

    if output:
        prediction = np.argmax(output["predictions"][0])
        confidence = output["predictions"][0][prediction]

        target_pipeline = joblib.load(directory / "target.joblib")
        classes = target_pipeline.named_transformers_["species"].categories_[0]

        result = {
            "prediction": classes[prediction],
            "confidence": confidence,
        }
    else:
        result = {"prediction": None}

    print(result)

    response_content_type = (
        "application/json" if context is None else context.accept_header
    )
    return json.dumps(result), response_content_type
