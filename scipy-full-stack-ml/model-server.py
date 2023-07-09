from metaflow import Flow, namespace
from fastapi import FastAPI
import pandas as pd
import numpy as np

FLOW_NAME = 'TitanicSurvivalPredictor'

# Assume `load-models` endpoint gets hit at least once.
champ = None; champ_cols = None
challenger = None; challenger_cols = None

# Create FastAPI instance.
api = FastAPI()

# How to respond to HTTP GET at / route.
@api.get("/")
def root():
    return {"message": "Hello there!"}

# Wrapper for baseline model to mimic sklearn.
class MajorityClassPredictor:
    def predict(self, X):
        return [0]

# This is bad to maintain twice, as it is copy-pasted from the XGBoost flow. 
# At scale / in practice this is the use case for a proper feature store.
def featurize(df):
    TARGET = 'Survived' # this wouldn't exist in real production scenario, it is what we want to predict!
    IGNORE_COLS = ['Name', 'Ticket']
    CATEGORICALS = ['Sex', 'Cabin', 'Embarked', 'Pclass', 'SibSp', 'Parch'] 
    df = pd.get_dummies(df, columns = CATEGORICALS)
    return df.drop(columns=[TARGET] + IGNORE_COLS), df[TARGET]

@api.get("/load-models")
def load_models(champ_namespace=None, challenger_namespace=None):
    "Set the objects for each model type. This is only intended as a proof of concept."

    global champ
    global challenger

    global champ_cols
    global challenger_cols

    # Set up champ.
    namespace(champ_namespace)
    run = Flow(FLOW_NAME).latest_successful_run
    model_type = run.data.model_type

    if model_type == 'baseline':
        champ = MajorityClassPredictor()
        champ_cols = None

    elif model_type == 'xgboost':
        champ = run.data.model
        champ_cols = list(run.data.cols)

    msg = f'Running {model_type} model as champion.'

    # Set up challenger.
    if challenger_namespace is not None:

        namespace(challenger_namespace)
        run = Flow(FLOW_NAME).latest_successful_run
        model_type = run.data.model_type

        if model_type == 'baseline':
            challenger = MajorityClassPredictor()
            challenger_cols = None

        elif model_type == 'xgboost':
            challenger = run.data.model
            challenger_cols = list(run.data.cols)

        msg += f'\nRunning {model_type} model as challenger.'
    else:
        print("No challenger model specified.")
    
    return msg

# How to respond to HTTP GET at /sentiment route.
@api.get("/get-pred")
def get_pred(data, which_model=None):

    features, _ = featurize(pd.read_json(data))

    if which_model is None:
        
        print("No model selected, randomly selected one with 4/5 chance of using champion.")

        if np.random.random() > 0.2: # send 80% of traffic to champ, 20% to challenger
            if champ_cols is not None:
                features = features.reindex(features.columns.union(champ_cols, sort=False), axis=1, fill_value=0)
            pred = champ.predict(features)[0]
            model_used = 'champ'
        else:
            if challenger_cols is not None:
                features = features.reindex(features.columns.union(challenger_cols, sort=False), axis=1, fill_value=0)
            pred = challenger.predict(features)[0]
            model_used = 'challenger'
        
    elif which_model == 'champion':
        if champ_cols is not None:
            features = features.reindex(features.columns.union(champ_cols, sort=False), axis=1, fill_value=0)
        pred = champ.predict(features)[0]
        model_used = 'champ'
        
    elif which_model == 'challenger':
        if challenger_cols is not None:
            features = features.reindex(features.columns.union(challenger_cols, sort=False), axis=1, fill_value=0)
        pred = challenger.predict(features)[0]
        model_used = 'challenger'

    # fastAPI doesn't deal with numpy types
    if isinstance(pred, np.int64):
        pred = pred.item()

    if pred not in [0,1]:
        print(f"{model_used} model is going rogue, and not predicting a 0 or 1.")
        print("Defaulting to always predict 0 strategy.")
        pred = 0

    print("\n\n PREDICTION: {} \n\n".format(pred))

    return {"prediction": pred, "model_used": model_used}