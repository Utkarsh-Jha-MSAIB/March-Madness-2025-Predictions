import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def build_model_input(mm_data: pd.DataFrame, macro_data: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a balanced input dataset for binary classification models
    by generating both original and mirrored matchups with proper labeling.
    Includes macro-level season merge.

    Parameters:
    ----------
    mm_data : pd.DataFrame
        Raw match-level statistics with features and scores

    macro_data : pd.DataFrame
        Season-level macro data to be merged on 'season'

    Returns:
    -------
    pd.DataFrame
        A balanced, model-ready DataFrame
    """

    # Step 1: Create win label
    mm_data['team1_win'] = (mm_data['team1_score'] > mm_data['team2_score']).astype(int)

    # Step 2: Select columns
    training_columns = [
        'team1_id','team2_id','team1_seed','team2_seed','game_id','team1_score','team2_score',
        'point_diff','team1_AdjEM', 'team2_AdjEM', 'SeedDiff', 'team1_eFG',
        'team2_eFG', 'TurnoverMargin','team1_FTR', 'team2_FTR','diff_dist','team1_win',
        'slot', 'season',
        'team1_blockpct','team1_oppfg2pct','team1_oppfg3pct','team1_oppftpct','team1_oppblockpct',
        'team1_f3grate','team1_oppf3grate','team1_arate','team1_opparate',
        'team2_blockpct','team2_oppfg2pct','team2_oppfg3pct','team2_oppftpct','team2_oppblockpct',
        'team2_f3grate','team2_oppf3grate','team2_arate','team2_opparate',
        'team1_tempo','team1_adjtempo','team1_oe','team1_de',
        'team2_tempo','team2_adjtempo','team2_oe','team2_de',
        'team1_coach_experience_score','team2_coach_experience_score'
    ]

    # Step 3: Sample split
    mm_data_1 = mm_data.sample(frac=.5, random_state=15)[training_columns]
    mm_data_2 = mm_data[~mm_data.index.isin(mm_data_1.index)][training_columns].reset_index(drop=True)
    mm_data_1 = mm_data_1.reset_index(drop=True)

    # Step 4: Flip team1 and team2
    mm_data_2['team1_win'] = 0
    mm_data_2['diff_dist'] *= -1
    mm_data_2['SeedDiff'] *= -1
    mm_data_2['TurnoverMargin'] *= -1
    mm_data_2['game_id'] = (
        mm_data_2['game_id'].str.split('-', expand=True)[0] + '-' +
        mm_data_2['game_id'].str.split('-', expand=True)[2] + '-' +
        mm_data_2['game_id'].str.split('-', expand=True)[1]
    )

    mm_data_2.columns = [
        'team2_id','team1_id','team2_seed','team1_seed','game_id','team2_score','team1_score',
        'point_diff','team2_AdjEM', 'team1_AdjEM', 'SeedDiff', 'team2_eFG',
        'team1_eFG', 'TurnoverMargin','team2_FTR', 'team1_FTR','diff_dist','team1_win',
        'slot', 'season',
        'team2_blockpct','team2_oppfg2pct','team2_oppfg3pct','team2_oppftpct','team2_oppblockpct',
        'team2_f3grate','team2_oppf3grate','team2_arate','team2_opparate',
        'team1_blockpct','team1_oppfg2pct','team1_oppfg3pct','team1_oppftpct','team1_oppblockpct',
        'team1_f3grate','team1_oppf3grate','team1_arate','team1_opparate',
        'team2_tempo','team2_adjtempo','team2_oe','team2_de',
        'team1_tempo','team1_adjtempo','team1_oe','team1_de',
        'team2_coach_experience_score','team1_coach_experience_score'
    ]

    # Step 5: Concatenate in correct order
    mm_train = pd.concat([mm_data_2, mm_data_1]).reset_index(drop=True)

    # Step 6: Merge macro data
    mm_train = pd.merge(mm_train, macro_data, how='left', on='season')

    return mm_train


def perform_pca_analysis(df: pd.DataFrame, drop_low_variance: bool = True, top_n: int = 3, plot: bool = True) -> dict:
    """
    Performs PCA analysis on numeric columns and identifies the top contributing variables
    for the first 4 principal components.

    Parameters:
    ----------
    df : pd.DataFrame
        The input data (e.g., from build_model_input)

    drop_low_variance : bool
        Whether to drop near-zero variance columns before PCA

    top_n : int
        Number of top contributing variables to print for each component

    plot : bool
        Whether to show correlation heatmap and explained variance plot

    Returns:
    -------
    dict
        Dictionary of top_n variable names per principal component (PC1–PC4)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # 1. Numeric only
    numeric_df = df.select_dtypes(include=[np.number])

    # 2. Drop low-variance cols
    if drop_low_variance:
        low_variance_cols = numeric_df.var()[numeric_df.var() < 0.01].index.tolist()
        print("Dropped low variance columns:", low_variance_cols)
        numeric_df = numeric_df.drop(columns=low_variance_cols)

    # 3. Correlation heatmap
    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

    # 4. Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # 5. Fit PCA
    pca = PCA()
    pca.fit(scaled_data)

    # 6. Explained variance plot
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by PCA Components')
        plt.grid(True)
        plt.show()

    # 7. Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(numeric_df.columns))],
        index=numeric_df.columns
    )

    # 8. Print top contributors
    for i in range(1, 5):
        component = f'PC{i}'
        print(f"\nTop variable contributions to {component}:")
        print(loadings[component].sort_values(ascending=False).head(top_n))

    # 9. Extract top contributors per PC
    important_vars = {}
    for i in range(1, 5):
        component = f'PC{i}'
        top_vars = loadings[component].abs().sort_values(ascending=False).head(top_n).index.tolist()
        important_vars[component] = top_vars

    print("\nRecommended variables based on PCA analysis:")
    for pc, vars in important_vars.items():
        print(f"{pc}: {vars}")

    return important_vars


