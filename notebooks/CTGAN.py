#!/usr/bin/env python
# coding: utf-8

# Standard imports
import pandas as pd
from tqdm import tqdm

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

# SDV imports
from sdv.tabular import CTGAN, CopulaGAN, TVAE
from sdv.metrics.tabular import (BinaryDecisionTreeClassifier,
                                 BinaryAdaBoostClassifier,
                                 BinaryLogisticRegression,
                                 BinaryMLPClassifier,
                                 LogisticDetection,
                                 SVCDetection)

# ML imports
from sklearn.model_selection import train_test_split

class SDVInputDataset:
    '''
    filepath: str: path to real dataset

    real_dataframe: pd.DataFrame: real dataframe (already imported)

    Pass one or the other of filepath or real_dataframe
    - the latter will override the former
    !TODO: Raise error instead if both passed

    primary_key: str: colname to use as primary key (column where every value
                      must be unique)

    header: int: Row number to use in dataframe import from filename

    fields_to_anonymise: dict:
        key:val of field to anonymise (string) and string referance to one of
        the following fakers: https://faker.readthedocs.io/en/master/providers.html
        common options:
        'name' ; 'address' ; 'country' ; 'city' ; 'email' ; 'telephone'

    '''
    def __init__(self,
                 filepath=None,
                 real_dataframe=None,
                 primary_key=None,
                 header=None,
                 fields_to_anonymize=None
                 ):

        if filepath is not None:
            self.filepath = filepath
            self.dataset = pd.read_csv(self.filepath, header=header)

        if real_dataframe is not None:
            self.dataset = real_dataframe

        self.primary_key = primary_key
        self.fields_to_anonymize = fields_to_anonymize

        self.real_train, self.real_test = train_test_split(self.dataset,
                                                           test_size=0.2,
                                                           random_state=42)


class SDVOutputDataset:
    '''
    nrows: int: number of rows of data to create (i.e. length of
                synthetic dataframe)
    '''
    def __init__(self,
                 nrows,
                 ):

        self.nrows = nrows


class SDVUniversalParams:
    '''
    models to run: list of str: list containing any (lowercase) combination of
    - ctgan
    - copulagan
    - tvae

    epochs: int: number of epochs to train for
                 only affects ctgan and copulagan

    batch_size: int: size of batches for training. Must be divisible by 10.
                     only affects ctgan and copulagan

    generator_dim: tuple of ints: layer dimensions of generator network

    discriminator_dim: tuple of ints: layer dimensions of discriminator network

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
    input_dataset: SDVInputDataset object
    universal_params: SDVUniversalParams object

    Returns CTGAN model
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
    input_dataset: SDVInputDataset object
    universal_params: SDVUniversalParams object

    Returns CopulaGAN model
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
    input_dataset: SDVInputDataset object
    universal_params: SDVUniversalParams object

    Returns TVAE model
    '''
    model = TVAE(primary_key=input_dataset.primary_key,
                 anonymize_fields=input_dataset.fields_to_anonymize
                 )

    model.fit(input_dataset.dataset)

    return model

# Fit multiple models at once


def get_models(run_params, input_dataset):
    '''
    input_dataset: SDVInputDataset object
    run_params: SDVUniversalParams object

    Takes list of models to fit from run_params and returns fitted models
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
    models: dict: output of get_models
    output_dataset: SDVOutputDataset object

    Returns dict of synthetic datasets generated using models specified in
    SDVUniversalParams object that was passed to get_models
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
    Convenience function for generating synthetic datasets

    input_dataset: SDVInputDataset object
    run_params: SDVUniversalParams object
    output_dataset: SDVOutputDataset object
    '''

    models = get_models(run_params=run_params,
                        input_dataset=input_dataset)

    return generate_datasets(models, output_dataset)


# def sdv_dataset_synthesizer_from_saved_model(model_pkl,
#                                              output_dataset,
#                                              run_params):
#     '''
#     '''
#     raise NotImplementedError


def compare_methods_column_pairs(dict_of_dfs, col1, col2):
    '''
    Return scatterplot of specified columns
    dict_of_dfs: dict: dictionary where keys are names of dataframes and values
                       are pd.DataFrame objects
    col1: str: column name for x axis. Must be present in every dataframe.
    col2: str: column name for x axis. Must be present in every dataframe.
    '''
    fig, ax = plt.subplots(2, 2,
                           figsize=(8, 8),
                           sharex=True, sharey=True)
    for i, (name, df) in enumerate(dict_of_dfs.items()):
        df.plot(kind='scatter', x=col1, y=col2,
                title=name, ax=ax.flatten()[i])
    plt.tight_layout()


def check_formats(dict_of_dfs, input_dataset):
    '''
    Check that all synthetic dataframes have the same columns and datatypes
    as the real dataframe

    dict_of_dfs: dict: dictionary where keys are names of dataframes and values
                       are pd.DataFrame objects

    input_dataset: SDVInputDataset object

    '''
    for name, df in dict_of_dfs.items():
        if name != "real":
            print(f"Column names the same for {name}: {np.all(df.columns == input_dataset.dataset.columns)}")
            print(f"Data types the same for {name}: {np.all(df.dtypes == input_dataset.dataset.dtypes)}")
            print()


def corr_plot(dict_of_dfs):
    '''
    Generate comparison correlation plots for dataframes

    dict_of_dfs: dict: dictionary where keys are names of dataframes and values
                       are pd.DataFrame objects
    '''
    fig, ax = plt.subplots(2, 2,
                           figsize=(8, 8),
                           sharex=True, sharey=True)
    for i, (name, df) in enumerate(dict_of_dfs.items()):
        sns.heatmap(df.corr(), ax=ax.flatten()[i])
        ax.flatten()[i].set_title(name)
    plt.tight_layout()


def simple_metrics(dict_of_dfs, input_dataset):
    '''
    Compute simple metrics (logistic detection score and SVC detection score)
    for synthetic dataframes

    From SDV Documentation:
    "The metrics of this family evaluate how hard it is to distinguish
    the synthetic data from the real data by using a Machine Learning model.
    To do this, the metrics will shuffle the real data and synthetic data
    together with flags indicating whether the data is real or synthetic,
    and then cross validate a Machine Learning model that tries to predict
    this flag. The output of the metrics will be the 1 minus the average ROC
    AUC score across all the cross validation splits."

    dict_of_dfs: dict: dictionary where keys are names of dataframes and values
                       are pd.DataFrame objects

    input_dataset: SDVInputDataset object
    '''
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


def classifier_comparison(dict_of_dfs, target_col, input_dataset):
    '''
    From SDV Documentation:
    This family of metrics will evaluate whether it is possible to replace
    the real data with synthetic data in order to solve a Machine Learning
    Problem by learning a Machine Learning model on the synthetic data and
    then evaluating the score which it obtains when evaluated on the real data.

    dict_of_dfs: dict: dictionary where keys are names of dataframes and values
                       are pd.DataFrame objects

    input_dataset: SDVInputDataset object

    target_col: str: column containing feature of interest that is being aimed
                     for

    input_dataset: SDVInputDataset object

    !! POSSIBLE ISSUE
    The classifiers don't take arguments so not sure how to set seed

    '''
    classifier_dict = {
        "Decision Tree": BinaryDecisionTreeClassifier,
        "AdaBoost": BinaryAdaBoostClassifier,
        "Logistic Regression": BinaryLogisticRegression,
        "MLP Classifier": BinaryMLPClassifier
    }

    results = []

    for classifier_name, classifier in classifier_dict.items():

        for name, df in dict_of_dfs.items():
            if name != "real":
                # Split so we have same size of training data when comparing
                # synthetic performance and real performance
                syn_train, syn_test = train_test_split(df,
                                                       test_size=0.2,
                                                       random_state=42)

                result = classifier.compute(syn_train,
                                            input_dataset.real_test,
                                            target=target_col)
            else:
                result = classifier.compute(input_dataset.real_train,
                                            input_dataset.real_test,
                                            target=target_col)

            results.append({'Model': name,
                            "Classifier": classifier_name,
                            "Result": result})

    return pd.DataFrame(results)


def plot_classifier_metrics(classifier_metrics_df):
    '''
    Plot the output of classifier_comparison() or simple_metrics(),
    grouped by metric or classifier
    '''
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.barplot(x='Classifier', y='Result', hue='Model',
                data=classifier_metrics_df,
                ax=ax)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('1 - AUC ROC (Higher = Better)')
    plt.ylim((0, 1.1));
