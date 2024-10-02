# build a baseline logistic regression

import polars as pl
import polars.selectors as cs
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score
from sklearn.linear_model import LogisticRegression

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

# cap the person age to 75
def cap_age(df):
    ans = df.with_columns(pl.when(pl.col('person_age') > 75).then(pl.lit(75)).otherwise(pl.col('person_age')).alias('capped_age'))
    return(ans)

df_train_capage = cap_age(df = df_train_default)
df_test_capage = cap_age(df = df_test_default)

# log transform the person_income
def transform_income(df):
    ans = df.with_columns(log_income = pl.col('person_income').log())
    return(ans)

df_train_trincome = transform_income(df = df_train_capage)
df_test_trincome = transform_income(df = df_test_capage)

# cap person_emp_length to 25 max
def cap_employ_length(df):
    ans = df.with_columns(emp_length = pl.when(pl.col('person_emp_length') > 25).then(pl.lit(25)).otherwise(pl.col('person_emp_length')))
    return(ans)

df_train_caplen = cap_employ_length(df = df_train_trincome)
df_test_caplen = cap_employ_length(df = df_test_trincome)

# convert all values in loan_intent to lower_case
df_train_lc = df_train_caplen.with_columns(loan_intent = pl.col('loan_intent').str.to_lowercase())
df_test_lc = df_test_caplen.with_columns(loan_intent = pl.col('loan_intent').str.to_lowercase())

# drop the person_income, person_home_ownership columns
cln_train = df_train_lc.select(pl.exclude('person_income', 'person_home_ownership','loan_grade', 'person_emp_length','person_age','cb_person_default_on_file'))
cln_test = df_test_lc.select(pl.exclude('person_income', 'person_home_ownership','loan_grade','person_emp_length','person_age','cb_person_default_on_file'))

# log transform the non-binary cols
def transform_log(df):
    ans= df.with_columns(loan_amnt = pl.col('loan_amnt').log1p(),
                    loan_int_rate = pl.col('loan_int_rate').log1p(),
                    loan_percent_income = pl.col('loan_percent_income').log1p(),
                    cb_person_cred_hist_length = pl.col('cb_person_cred_hist_length').log1p(),
                    capped_age = pl.col('capped_age').log1p(),
                    emp_length = pl.col('emp_length').log1p())
    return(ans)

cln_train_log = transform_log(df = cln_train)
cln_test_log = transform_log(df = cln_test)

# create dummies
def create_dummy(df):
    ans = df.to_dummies(cs.by_dtype(pl.String))
    return(ans)

cln_train_dummy = create_dummy(df = cln_train_log)
cln_test_dummy = create_dummy(df = cln_test_log)

# cln_train_dummy.head()
# cln_train_dummy.select(cs.by_dtype(pl.Int64, pl.Float64)).select(pl.exclude('id','log_income','loan_status')).select(pl.all().log1p().skew())
# cln_train_dummy.select(cs.by_dtype(pl.Int64, pl.Float64)).select(pl.exclude('id','log_income','loan_status')).select(pl.all().log1p().kurtosis(fisher = False))
# cln_train_dummy.select('emp_length').describe()
# raw_train.filter(pl.col('person_emp_length') < 1).head()
# cln_train_dummy.select(cs.by_dtype(pl.Int64, pl.Float64)).select(pl.exclude('id','log_income','loan_status')).columns


# let's create a train/test split
xvars = cln_train_dummy.select(pl.exclude('loan_status')).columns
yvar = cln_train_dummy.select('loan_status').columns

# train/test split
x_train, x_valid, y_train, y_valid = train_test_split(cln_train_dummy.select(xvars), cln_train_dummy.select(yvar), random_state=1337, test_size = 0.3, stratify = cln_train_dummy.select('loan_status'))

# scale all features except dummies
scale_cols = x_train.select(cs.by_dtype(pl.Int64, pl.Float64)).select(pl.exclude('id')).columns
ss = StandardScaler()
nptr_scaled = ss.fit_transform(x_train.select(scale_cols))
dftr_scaled = pl.from_numpy(nptr_scaled, schema = scale_cols)

npvalid_scaled = ss.transform(x_valid.select(scale_cols))
dfva_scaled = pl.from_numpy(npvalid_scaled, schema = scale_cols)

xtrain_scaled = pl.concat([x_train.select(pl.exclude(scale_cols)), dftr_scaled], how = 'horizontal')
xvalid_scaled = pl.concat([x_valid.select(pl.exclude(scale_cols)), dfva_scaled], how = 'horizontal')

#logreg train
lr = LogisticRegression(penalty='l1', solver = 'liblinear')
lr_mdl = lr.fit(xtrain_scaled.select(pl.exclude('id')), pl.Series(y_train))

# make predictions
baseline_ypred = lr_mdl.predict(xvalid_scaled.select(pl.exclude('id')))
baseline_ypredprob = lr_mdl.predict_proba(xvalid_scaled.select(pl.exclude('id')))

# find the metrics for model performance
def score_model(model_name, y_test, y_pred):

    acc = accuracy_score(y_test, y_pred)
    kap = cohen_kappa_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    dict_metric = pl.DataFrame({'metric':['accuracy','kappa','auc'], model_name: [acc, kap, roc]})

    return dict_metric

score_model(model_name = 'baseline', y_test = y_valid, y_pred = baseline_ypred)

# see coefficients
pl.from_numpy(lr_mdl.coef_, schema=xtrain_scaled.select(pl.exclude('id')).columns).unpivot()

"""
┌──────────┬──────────┐
│ metric   ┆ baseline │
│ ---      ┆ ---      │
│ str      ┆ f64      │
╞══════════╪══════════╡
│ accuracy ┆ 0.895476 │
│ kappa    ┆ 0.482489 │
│ auc      ┆ 0.69952  │
└──────────┴──────────┘
"""
