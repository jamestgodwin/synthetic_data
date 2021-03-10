def synth_metrics(real_data, synthetic_data):
    '''
    Returns a group of metrics and plots to evaluate synthetic dataset

    Params:
    -------

    actual_data: pd.DataFrame
    The real data from which the synthetic data is derived from

    synthetic_data: pd.DataFrame
    The synthetic data

    Returns:
    --------
    '''
    fig, axes = plt.subplots(1,2, figsize=(20, 10))
    sns.heatmap(real_data.corr(), ax=axes[0], )
    sns.heatmap(synthetic_data.corr(), ax=axes[1])
    avg_diff = (synthetic_data.corr() - real_data.corr())\
        .abs().values.flatten().mean()
    plt.show()
    print(f'Average difference between correlations: {avg_diff:.3f}')
    num_cols = len(real_data.columns)
    fig, axes = plt.subplots((num_cols+1)//2,2, figsize=(num_cols*3, num_cols*3))
    for i, col in enumerate(real_data.columns):
        axes.flatten()[i].boxplot(back_pain[' pelvic_incidence  (numeric) '], positions=[1])
        axes.flatten()[i].boxplot(smote_df[' pelvic_incidence  (numeric) '], positions=[2])
    plt.show()
