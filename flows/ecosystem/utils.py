



def validate(df,name='flowers-test-flow-checkpoint'):
    """
    Validate data using great expectations
    """
    from ruamel import yaml
    import great_expectations as ge
    from great_expectations.core.batch import RuntimeBatchRequest

    context = ge.get_context()

    checkpoint_config = {
            "name": name,
            "config_version": 1,
            "class_name": "SimpleCheckpoint",
            "run_name_template": "%Y%m%d-%H%M%S-flower-power",
            "validations": [
                {
                    "batch_request": {
                        "datasource_name": "flowers",
                        "data_connector_name": "default_runtime_data_connector_name",
                        "data_asset_name": "iris",
                    },
                    "expectation_suite_name": "flowers-testing-suite",
                }
            ],
        }
    context.add_checkpoint(**checkpoint_config)


    results = context.run_checkpoint(
            checkpoint_name=name,
            batch_request={
                "runtime_parameters": {"batch_data": df},
                "batch_identifiers": {
                    "default_identifier_name": "<YOUR MEANINGFUL IDENTIFIER>"
                },
            },
        )
    context.build_data_docs()
    context.open_data_docs()