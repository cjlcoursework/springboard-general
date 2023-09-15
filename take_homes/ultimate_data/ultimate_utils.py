from typing import Set, Any

import pandas as pd
import numpy as np
import shap
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from woodwork.logical_types import Categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_curve
from typing import Set, Any

from sklearn.model_selection import train_test_split
from xgboost import DMatrix
import xgboost as xgb


def read_ultimate_data():
    return pd.read_json('ultimate_data_challenge.json',
                        convert_dates=['last_trip_date', 'signup_date'],
                        dtype={
                            'city': 'category',
                            'phone': 'category'
                        }
                        )


def add_retained_label(df):
    from datetime import timedelta

    # Calculate the maximum last_trip_date
    max_last_trip_date = df['last_trip_date'].max()

    # Calculate the date 30 days ago from the maximum last_trip_date
    cutoff_date = max_last_trip_date - timedelta(days=30)

    df['retained'] = np.where(df['last_trip_date'] > cutoff_date, 1, 0)


def perform_data_transforms(df):
    # cast to floats
    df['trips_in_first_30_days'] = df['trips_in_first_30_days'].astype(float)

    df['signup_month'] = df['signup_date'].dt.month
    df['signup_month'] = df['signup_month'].astype('category')

    df['last_trip_month'] = df['last_trip_date'].dt.month  # --- this is the top feature, but makes no sense so remove it?
    df['last_trip_month'] = df['last_trip_month'].astype('category')

    # and maybe the length of time from signup to last_trip could have some value as a feature
    df['last_trip_days_since_su'] = (df['last_trip_date'] - df['signup_date']).dt.days

    # change booleans to int
    df['ultimate_black_user'] = df['ultimate_black_user'].astype(int)

    # lower case the phone names and fill the missing phone cells with 'other'
    df['phone'] = np.where(df['phone'].isna(), 'other', df['phone'].str.lower())
    df['phone'] = df['phone'].astype('category')

    # just cosmetically, make the column names lower case with no spaces in the name
    df['city'] = df['city'].str.replace(' ', '_').str.lower().astype('category')

    # fill the null avg_ columns with the mean - at least for feature selection
    df['avg_rating_by_driver'].fillna(df['avg_rating_by_driver'].mean(), inplace=True)
    df['avg_rating_of_driver'].fillna(df['avg_rating_of_driver'].mean(), inplace=True)


def correlate_to_target(df: pd.DataFrame, target_column: str, top_n: int) -> (pd.DataFrame, Set):
    # Calculate correlation matrix
    correlation_matrix = df.corr()

    # Filter and sort correlation coefficients by absolute value
    corr_df = pd.DataFrame(correlation_matrix.abs().unstack().sort_values(ascending=False))
    corr_df.reset_index(inplace=True)
    corr_df.columns = ['x', 'y', 'corr']

    not_self_correlated = (corr_df.y != target_column)
    win_correlated = (corr_df.x == target_column)

    df = corr_df.loc[win_correlated & not_self_correlated] \
        .sort_values(by='corr', ascending=False) \
        .drop(columns=['x']) \
        .head(top_n) \
        .copy()

    s = set(df['y'].values)

    return df, s


def plot_correlations(scores: Any, features: Any, title: str):
    # Plot feature importance
    plt.figure(figsize=(10, 10))
    plt.barh(range(len(scores)), scores, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Score')
    plt.ylabel('Column')
    plt.title(title)
    plt.show()


def plot_heatmap(df, cmap='coolwarm', label_fontsize=10):  # Add a parameter for x label font size
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 10))
    plt.imshow(correlation_matrix, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.title('Heatmap')

    # Set x-axis and y-axis labels with fontsize
    plt.xticks(np.arange(len(numeric_df.columns)), numeric_df.columns, rotation=45, fontsize=label_fontsize)
    plt.yticks(np.arange(len(numeric_df.columns)), numeric_df.columns, fontsize=label_fontsize)

    plt.savefig('temp.png', dpi=300, bbox_inches='tight')
    plt.show()


def perform_featuretools_dfs(features, target):
    import featuretools as ft

    es = ft.EntitySet(id="my_entityset")

    es = es.add_dataframe(
        dataframe_name="features",
        dataframe=features,
        index='index_col',
        time_index="signup_date",
        logical_types={
            "city": Categorical,
            "phone": Categorical,
            "signup_month": Categorical

        },
    )
    es = es.add_dataframe(
        dataframe_name="target",
        dataframe=target,
        index='child_index'
    )

    es = es.add_relationship(
        "features", "index_col", "target", "index_col"
    )
    return ft.dfs(entityset=es, target_dataframe_name="target")


def calc_feature_importance(X: pd.DataFrame, y: pd.Series, top_n=30) -> (pd.DataFrame, Set):
    # Create an XGBoost model
    model = xgb.XGBRegressor()

    # Fit the model
    model.fit(X, y)

    # Get feature importance scores
    importance_scores = model.feature_importances_

    # Sort feature importance scores
    sorted_indices = importance_scores.argsort()[::-1]
    sorted_scores = importance_scores[sorted_indices]
    feature_names = X.columns[sorted_indices]

    # Get the top 'n' feature importance scores and names
    top_scores = sorted_scores[:top_n]
    top_features = feature_names[:top_n]
    df = pd.DataFrame({'y': top_features, 'corr': top_scores})

    set_xg_cols = set(top_features)
    return df, set(top_features)


def perform_xgboost_selection(X, y, top_n=30):
    # Create a DMatrix for X_train and X_test with categorical features enabled
    X_dmatrix = DMatrix(X, label=y, enable_categorical=True)

    # Create and train an XGBoost model
    params = {
        'objective': 'binary:logistic',  # for binary classification
        'eval_metric': 'logloss',  # you can change the evaluation metric
        'random_state': 42  # set a random state for reproducibility
    }

    model = xgb.train(params, X_dmatrix, num_boost_round=100)

    # Get feature importance scores from the Booster model
    importance_scores = model.get_score(importance_type='weight')  # You can use other importance types as well

    # Convert the importance scores dictionary into a DataFrame
    df_importance = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['importance'])

    # Sort the features by importance in descending order
    df_importance = df_importance.sort_values(by='importance', ascending=False)

    # Get the top 30 feature names and their importance scores
    top_features = df_importance.head(top_n)

    return top_features


def encode_ultimate_features(df):
    continuous_columns = [
        'avg_rating_by_driver',
        'avg_rating_of_driver',
        'avg_surge',
        'surge_pct',
        'avg_dist',
        'weekday_pct',
        'trips_in_first_30_days',
        'last_trip_days_since_su'
    ]

    # category_columns = ['phone', 'city', 'signup_month', 'last_trip_month']
    category_columns = ['phone', 'city', 'signup_month', 'last_trip_month']

    # Perform one-hot encoding on the cat columns
    df_encoded = pd.get_dummies(df, columns=category_columns)

    # Apply MinMax scaling to the numeric columns
    scaler = StandardScaler()
    df_encoded[continuous_columns] = scaler.fit_transform(df_encoded[continuous_columns])

    df_encoded.drop(columns=['signup_date', 'last_trip_date'], inplace=True)

    return df_encoded


def print_f1_scores(y_pred, y_test):
    # Calculate precision, recall, and F1 scores
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


def plot_accuracy(r):
    # Evaluate the model
    plt.plot(r.history['accuracy'], label='accuracy')
    plt.plot(r.history['val_accuracy'], label='val-accuracy')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_pred, y_test):
    cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot(cmap='Blues')
    plt.show()


def plot_loss(r):
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val-loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


def plot_roc_curve(y_actual, y_predicted):

    # Assuming y_predicted and y_actual are the predicted and actual labels
    fpr, tpr, thresholds = roc_curve(y_actual, y_predicted)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()


def shap_explainer(model, X):

    # Assuming you have a trained Keras model 'model' and a dataset 'X'
    explainer = shap.DeepExplainer(model, data=X)
    shap_values = explainer.shap_values(X)

    # Plotting the SHAP values
    shap.summary_plot(shap_values, X)