import polars as pl
import polars.selectors as cs
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier

# improve print outputs from polars
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(3000)
pl.Config.set_tbl_cols(-1)

# load the train & test dataset
raw_train = pl.read_csv('data/train.csv')
raw_test = pl.read_csv('data/test.csv')

# function to reduce the categories
def reduce_cats(df):
    ans = df \
        .with_columns(pl.when(pl.col('person_home_ownership') != 'RENT').then(pl.lit(0)).otherwise(pl.lit(1)).alias('renting')) \
        .with_columns(pl.when(pl.col('loan_grade') == 'A').then(pl.lit('a')).when(pl.col('loan_grade')=='B').then(pl.lit('b')).otherwise(pl.lit('other')).alias('new_loan_grade'))

    return(ans)

df_train_lowcat = reduce_cats(df = raw_train)
df_test_lowcat = reduce_cats(df = raw_test)

# function to change cb_person_default_on_file from Y/N to 0,1
def change_default(df):
    ans = df.with_columns(pl.when(pl.col('cb_person_default_on_file') == 'N').then(pl.lit(0)).otherwise(pl.lit(1)).alias('default'))
    return(ans)

df_train_default = change_default(df = df_train_lowcat)
df_test_default = change_default(df = df_test_lowcat)

# convert all values in loan_intent to lower_case
df_train_lc = df_train_default.with_columns(loan_intent = pl.col('loan_intent').str.to_lowercase())
df_test_lc = df_test_default.with_columns(loan_intent = pl.col('loan_intent').str.to_lowercase())

# drop the person_income, person_home_ownership columns
cln_train = df_train_lc.select(pl.exclude('person_home_ownership','loan_grade','cb_person_default_on_file'))
cln_test = df_test_lc.select(pl.exclude('person_home_ownership','loan_grade','cb_person_default_on_file'))

# create dummies
def create_dummy(df):
    ans = df.to_dummies(cs.by_dtype(pl.String))
    return(ans)

cln_train_dummy = create_dummy(df = cln_train)
cln_test_dummy = create_dummy(df = cln_test)


# let's create a train/test split
xvars = cln_train_dummy.select(pl.exclude('loan_status')).columns
yvar = cln_train_dummy.select('loan_status').columns

# train/test split
x_train, x_valid, y_train, y_valid = train_test_split(cln_train_dummy.select(xvars), cln_train_dummy.select(yvar), random_state=1337, test_size = 0.3, stratify = cln_train_dummy.select('loan_status'))

rf = RandomForestClassifier(n_estimators=250, n_jobs=-1)
rf.fit(x_train.select(pl.exclude('id')), pl.Series(y_train))

# make predictions
baseline_ypred = rf.predict(x_valid.select(pl.exclude('id')))
baseline_ypredprob = rf.predict_proba(x_valid.select(pl.exclude('id')))

# find the metrics for model performance
def score_model(model_name, y_test, y_pred):

    acc = accuracy_score(y_test, y_pred)
    kap = cohen_kappa_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    dict_metric = pl.DataFrame({'metric':['accuracy','kappa','auc'], model_name: [acc, kap, roc]})

    return dict_metric

score_model(model_name = 'baseline', y_test = y_valid, y_pred = baseline_ypred)

""""
┌──────────┬──────────┐
│ metric   ┆ baseline │
│ ---      ┆ ---      │
│ str      ┆ f64      │
╞══════════╪══════════╡
│ accuracy ┆ 0.944413 │
│ kappa    ┆ 0.746845 │
│ auc      ┆ 0.835753 │
└──────────┴──────────┘
"""

# variable importance plot
tbl_varimp = pl.DataFrame({'features': list(x_train.select(pl.exclude('id')).columns), 'importance': rf.feature_importances_}).sort(by = ['importance'], descending = True).head(10)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.barh(tbl_varimp['features'], tbl_varimp['importance'])
ax.set_title("Variable Importance Plot")
ax.invert_yaxis()
caption1 = "Higher bars are better."
fig.text(0.75, 0.40, caption1, ha = 'center', color = 'darkblue')
fig.tight_layout()
plt.show()