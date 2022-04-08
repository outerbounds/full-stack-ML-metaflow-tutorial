import argparse
import joblib
import os
import numpy as np
import json


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model/model.joblib"))



# def input_fn(request_body, request_content_type):
#     if request_content_type == 'text/csv':
#         samples = []
#         for r in request_body.split():
#             samples.append(list(map(float,r.split(','))))
#         return np.array(samples)
#     else:
#         raise ValueError("Thie model only supports text/csv input")



# def output_fn(prediction, content_type):
#     return '|'.join([t for t in prediction])


# def input_handler(data, context):
#     """ Pre-process request input before it is sent to TensorFlow Serving REST API
#     Args:
#         data (obj): the request data, in format of dict or string
#         context (Context): an object containing request and configuration details
#     Returns:
#         (dict): a JSON-serializable dict that contains request body and headers
#     """
#     if context.request_content_type == 'application/json':
#         # pass through json (assumes it's correctly formed)
#         d = data.read().decode('utf-8')
#         return d if len(d) else ''

#     if context.request_content_type == 'text/csv':
#         # very simple csv handler
#         return json.dumps({
#             'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
#         })

#     raise ValueError('{{"error": "unsupported content type {}"}}'.format(
#         context.request_content_type or "unknown"))


# def output_handler(data, context):
#     """Post-process TensorFlow Serving output before it is returned to the client.
#     Args:
#         data (obj): the TensorFlow serving response
#         context (Context): an object containing request and configuration details
#     Returns:
#         (bytes, string): data to return to client, response content type
#     """
#     if data.status_code != 200:
#         raise ValueError(data.content.decode('utf-8'))

#     response_content_type = context.accept_header
#     prediction = data.content
#     return prediction, response_content_type