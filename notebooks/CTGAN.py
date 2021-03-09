#!/usr/bin/env python
# coding: utf-8

from sdv.tabular import CTGAN, CopulaGAN, TVAE
import pandas as pd
from tqdm import tqdm

class SDVInputDataset:
    '''
    !! TODO: Add option to pass df instead of filename

    fields_to_anonymise: dict:
        key:val of field to anonymise (string) and string referance to one of
        the following fakers: https://faker.readthedocs.io/en/master/providers.html
        common options:
        'name' ; 'address' ; 'country' ; 'city' ; 'email' ; 'telephone'

    '''
    def __init__(self,
                 filepath,
                 primary_key=None,
                 header=None,
                 fields_to_anonymize=None
                 ):

        self.filepath = filepath
        self.primary_key = primary_key
        self.fields_to_anonymize = fields_to_anonymize

        self.dataset = pd.read_csv(self.filepath, header=header)


class SDVOutputDataset:
    '''
    '''
    def __init__(self,
                 nrows,
                ):

        self.nrows = nrows


class SDVUniversalParams:
    '''
    '''
    def __init__(self,
                 models_to_run=['ctgan'],
                 epochs=300,
                 batch_size=500):

        self.models_to_run = models_to_run
        self.epochs = epochs

        if batch_size % 10 != 0:
            print("Batch size invalid. Must be divisible by 10. Setting "
                   + "to default (500)")
            self.batch_size=500
        else:
            self.batch_size=batch_size


def fit_ctgan(input_dataset):
    '''
    '''
    model = CTGAN(primary_key=input_dataset.primary_key,
                  anonymize_fields=input_dataset.fields_to_anonymize
                 )

    model.fit(input_dataset.dataset)

    return model

def fit_copulagan(input_dataset):
    '''
    '''
    model = CopulaGAN(primary_key=input_dataset.primary_key,
                      anonymize_fields=input_dataset.fields_to_anonymize
                      )

    model.fit(input_dataset.dataset)

    return model


def fit_tvae(input_dataset):
    '''
    '''
    model = TVAE(primary_key=input_dataset.primary_key,
                 anonymize_fields=input_dataset.fields_to_anonymize
                 )

    model.fit(input_dataset.dataset)

    return model


def get_models(run_params, input_dataset):
    '''
    '''
    models = {}

    if 'ctgan' in run_params.models_to_run:
        models['ctgan'] = fit_ctgan(input_dataset)

    if 'copulagan' in run_params.models_to_run:
        models['copulagan'] = fit_copulagan(input_dataset)

    if 'tvae' in run_params.models_to_run:
        models['tvae'] = fit_tvae(input_dataset)

    return models


def generate_datasets(models, output_dataset):
    '''

    '''

    synthetic_datasets = {}

    for model_name, model in tqdm(models.items()):
        print(f"Running for {model_name}")
        synthetic_datasets[model_name] = model.sample(output_dataset.nrows)

    return synthetic_datasets



def sdv_dataset_synthesizer(input_dataset,
                            output_dataset,
                            run_params):
    '''
    TODO: IMPLEMENT RUNPARAMS
    '''

    models = get_models(run_params=run_params,
                        input_dataset=input_dataset)


    return generate_datasets(models, output_dataset)

def sdv_dataset_synthesizer_from_saved_model(model_pkl,
                                             output_dataset,
                                             run_params):
    '''
    '''
    raise NotImplementedError
