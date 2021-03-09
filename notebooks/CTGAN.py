#!/usr/bin/env python
# coding: utf-8

from sdv.tabular import CTGAN, CopulaGAN, TVAE
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sdv.metrics.tabular import (BinaryDecisionTreeClassifier,
                                 BinaryAdaBoostClassifier,
                                 BinaryLogisticRegression,
                                 BinaryMLPClassifier,
                                 LogisticDetection,
                                 SVCDetection)
from sklearn.model_selection import train_test_split


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
                 batch_size=500,
                 generator_dim=(256, 256),
                 discriminator_dim=(256, 256),
                 ):

        self.models_to_run = models_to_run
        self.epochs = epochs

        if batch_size % 10 != 0:
            print("Batch size invalid. Must be divisible by 10. Setting "
                  + "to default (500)")
            self.batch_size = 500
        else:
            self.batch_size = batch_size

        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim

# Model fitting functions


def fit_ctgan(input_dataset, universal_params):
    '''
    '''
    model = CTGAN(primary_key=input_dataset.primary_key,
                  anonymize_fields=input_dataset.fields_to_anonymize,
                  epochs=universal_params.epochs,
                  batch_size=universal_params.batch_size,
                  generator_dim=universal_params.generator_dim,
                  discriminator_dim=universal_params.discriminator_dim,
                 )

    model.fit(input_dataset.dataset)

    return model


def fit_copulagan(input_dataset, universal_params):
    '''
    '''
    model = CopulaGAN(primary_key=input_dataset.primary_key,
                      anonymize_fields=input_dataset.fields_to_anonymize,
                      epochs=universal_params.epochs,
                      batch_size=universal_params.batch_size,
                      generator_dim=universal_params.generator_dim,
                      discriminator_dim=universal_params.discriminator_dim,
                      )

    model.fit(input_dataset.dataset)

    return model


def fit_tvae(input_dataset, universal_params):
    '''
    '''
    model = TVAE(primary_key=input_dataset.primary_key,
                 anonymize_fields=input_dataset.fields_to_anonymize
                 )

    model.fit(input_dataset.dataset)

    return model

# Fit multiple models at once


def get_models(run_params, input_dataset):
    '''
    '''
    models = {}

    if 'ctgan' in run_params.models_to_run:
        models['ctgan'] = fit_ctgan(input_dataset, run_params)

    if 'copulagan' in run_params.models_to_run:
        models['copulagan'] = fit_copulagan(input_dataset, run_params)

    if 'tvae' in run_params.models_to_run:
        models['tvae'] = fit_tvae(input_dataset, run_params)

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


def compare_methods_column_pairs(dict_of_dfs, col1, col2):
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    for i, (name, df) in enumerate(dict_of_dfs.items()):
        df.plot(kind='scatter', x=col1, y=col2,
                title=name, ax=ax.flatten()[i])
    plt.tight_layout()


def check_formats(dict_of_dfs, input_dataset):
    "Comparing with real dataframe"
    for name, df in dict_of_dfs.items():
        if name != "real":
            print(f"Column names the same for {name}: {np.all(df.columns == input_dataset.dataset.columns)}")
            print(f"Data types the same for {name}: {np.all(df.dtypes == input_dataset.dataset.dtypes)}")
            print()


def corr_plot(dict_of_dfs):
    fig, ax = plt.subplots(2,2, figsize=(8,8), sharex=True, sharey=True)
    for i, (name, df) in enumerate(dict_of_dfs.items()):
        sns.heatmap(df.corr(), ax=ax.flatten()[i])
        ax.flatten()[i].set_title(name)
    plt.tight_layout()


def simple_metrics(dict_of_dfs, input_dataset):
    results = []

    metrics_dict = {"Logistic Detection Score": LogisticDetection,
                    "SVC Detection Score": SVCDetection}

    for metric_name, metric in metrics_dict.items():

        for name, df in dict_of_dfs.items():
            result = LogisticDetection.compute(input_dataset.dataset, df)

            results.append({'Model': name,
                            "Classifier": metric_name,
                            "Result": result})
    return pd.DataFrame(results)


def classifier_comparison(dict_of_dfs, target_col):
    classifier_dict = {
        "Decision Tree": BinaryDecisionTreeClassifier,
        "AdaBoost": BinaryAdaBoostClassifier,
        "Logistic Regression": BinaryLogisticRegression,
        "MLP Classifier": BinaryMLPClassifier
    }

    results = []

    real_train, real_test = train_test_split(dict_of_dfs['real'],
                                             test_size=0.2,
                                             random_state=42)

    for classifier_name, classifier in classifier_dict.items():

        for name, df in dict_of_dfs.items():
            if name != "real":
                # Split so we have same size of training data when comparing
                # synthetic performance and real performance
                syn_train, syn_test = train_test_split(df,
                                                       test_size=0.2,
                                                      random_state=42)

                result = classifier.compute(syn_train, real_test,
                                            target=target_col)
            else:
                result = classifier.compute(real_train, real_test,
                                            target=target_col)
            results.append({'Model': name,
                            "Classifier": classifier_name,
                            "Result": result})
    return pd.DataFrame(results)


def plot_classifier_metrics(classifier_metrics_df):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.barplot(x='Classifier', y='Result', hue='Model',
                data=classifier_metrics_df,
                ax=ax)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('1 - AUC ROC (Higher = Better)')
    plt.ylim((0, 1.1));
