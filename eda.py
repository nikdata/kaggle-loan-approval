# libraries ----
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

# improve print outputs from polars
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(3000)
pl.Config.set_tbl_cols(-1)


# load files
raw_train = pl.read_csv('data/train.csv')
raw_test = pl.read_csv('data/test.csv')

# quick preview ----
raw_train.shape
raw_train.glimpse()
# 58645 rows, 13 columns
# response var is loan_status
# 6 integer columns, 3 float, 4 string

raw_test.shape
raw_test.glimpse()
# 39098 rows, 12 columns
# response var "loan_status" is missing

# CHECK FOR MISSING VALUES ----
raw_train.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
raw_test.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
# no missing values

# SKEW ----
raw_train.select(cs.by_dtype(pl.Int64, pl.Float64)).select(pl.all().skew(bias=False).round(3))
raw_test.select(cs.by_dtype(pl.Int64, pl.Float64)).select(pl.all().skew(bias=False).round(3))
# a skew of 0 means normal distribution
# most columns in train/test have similar skew values
# skew of person_emp_length in train is much higher than test
# many columns show bell shape

# KURT ----
raw_train.select(cs.by_dtype(pl.Int64, pl.Float64)).select(pl.all().kurtosis(fisher=False).round(3))
raw_test.select(cs.by_dtype(pl.Int64, pl.Float64)).select(pl.all().kurtosis(fisher=False).round(3))
# a kurt value of 3 means normal/no skew
# loan_percent_income seems to have limited skew
# person_emp_length has significantly different kurt in train data vs. test; all others similar kurtosis

# ANALYZE STRINGS - UNIQUE COUNT ----
raw_train.select(cs.by_dtype(pl.String)).select(pl.all().n_unique())
raw_test.select(cs.by_dtype(pl.String)).select(pl.all().n_unique())
# both have same number of unique values
# loan_intent & loan_grade have more than 5 unique values: consider consolidating
# person_home_ownership has 4 unique values
# cb_person_default_on_file is a binary column (Y/N only)

# let's confirm if both have the exact same values
raw_train.select('person_home_ownership').unique().sort(by = 'person_home_ownership') == raw_test.select('person_home_ownership').unique().sort(by = 'person_home_ownership')
raw_train.select('loan_intent').unique().sort(by = 'loan_intent') == raw_test.select('loan_intent').unique().sort(by = 'loan_intent')
raw_train.select('loan_grade').unique().sort(by = 'loan_grade') == raw_test.select('loan_grade').unique().sort(by = 'loan_grade')
raw_train.select('cb_person_default_on_file').unique().sort(by = 'cb_person_default_on_file') == raw_test.select('cb_person_default_on_file').unique().sort(by = 'cb_person_default_on_file')
# confirmed: both train & test have the same string values

# what are the unique values
raw_train.select('person_home_ownership').unique()
raw_train.select('loan_intent').unique()
raw_train.select('loan_grade').unique().sort(by = 'loan_grade')
raw_train.select('cb_person_default_on_file').unique()

# loan amount summary by loan grade
raw_train \
    .group_by('loan_grade', 'loan_status') \
    .agg(min_loanamt = pl.col('loan_amnt').min(),
         median_loanamt = pl.col('loan_amnt').median(),
         avg_loanamt = pl.col('loan_amnt').mean(),
         max_loanamt = pl.col('loan_amnt').max()) \
    .sort(by = 'loan_grade')

raw_train \
    .group_by('loan_grade', 'loan_status') \
    .agg(median_loanamt = pl.col('loan_amnt').median()) \
    .sort(by = 'loan_grade') \
    .pivot(on = 'loan_status', values = 'median_loanamt') \
    .select('loan_grade','0','1')

# about loan grades: https://blog.groundfloor.com/groundfloors-loan-grading-factors-explained#:~:text=Loan%20Grade,-The%20loan%20grade&text=For%20example%2C%20Grade%20A%20loans,but%20correspondingly%20higher%20interest%20rates.

raw_train.group_by('loan_grade').len().sort(by = 'loan_grade').with_columns(pct = (pl.col('len')/pl.col('len').sum() * 100).round(1))
raw_test.group_by('loan_grade').len().sort(by = 'loan_grade').with_columns(pct = (pl.col('len')/pl.col('len').sum() * 100).round(1))

raw_train.group_by('loan_intent').len().sort(by = 'loan_intent').with_columns(pct = (pl.col('len')/pl.col('len').sum() * 100).round(1))
raw_test.group_by('loan_intent').len().sort(by = 'loan_intent').with_columns(pct = (pl.col('len')/pl.col('len').sum() * 100).round(1))

raw_train.group_by('person_home_ownership').len().sort(by = 'person_home_ownership').with_columns(pct = (pl.col('len')/pl.col('len').sum() * 100).round(1))
raw_test.group_by('person_home_ownership').len().sort(by = 'person_home_ownership').with_columns(pct = (pl.col('len')/pl.col('len').sum() * 100).round(1))

raw_train.group_by('person_home_ownership','loan_status').len().sort(by = ['person_home_ownership','loan_status']).pivot(on = 'loan_status', values = 'len')

# let's see the number of cases by response var
raw_train.group_by('loan_status').len()

"""
PLOTS
"""

# histogram & boxplot of person age
# sns.set_theme()
# fig, ax = plt.subplots(1,1)
# p = so.Plot(raw_train, x = 'person_age').add(so.Bar(), so.Hist(stat = 'count', bins = 20)).label(title = 'Histogram: person_age')
# p.on(ax).show()
number_cols = raw_train.select(cs.by_dtype(pl.Int64, pl.Float64)).select(pl.exclude('id','loan_status')).columns
for i in range(0,len(number_cols)):
    fig, ax = plt.subplots(1, 1)
    p = so.Plot(raw_train, x=number_cols[i]).add(so.Bar(), so.Hist(stat='count', bins=20)).label(title='Histogram: ' + number_cols[i])
    p.on(ax).show()

# person age
fig, ax = plt.subplots()
p = so.Plot(raw_train.filter(pl.col('person_age') >= 60.0)).add(so.Bars(), so.Hist(stat = 'count', bins = 20), x = 'person_age').label(title = "age distribution")
p.on(ax).show()

raw_train.with_columns(rounded_age = ((pl.col('person_age')/10.0).floor()) * 10).group_by('rounded_age').len().sort(by = 'rounded_age').with_columns(pct = ((pl.col('len')/pl.col('len').sum())*100).round(1))
raw_test.with_columns(rounded_age = ((pl.col('person_age')/10.0).floor()) * 10).group_by('rounded_age').len().sort(by = 'rounded_age').with_columns(pct = ((pl.col('len')/pl.col('len').sum())*100).round(1))

# person income
fig, ax = plt.subplots()
p = so.Plot(raw_train).add(so.Bars(), so.Hist(stat = 'count', bins = 20), x = 'person_income').label(title = "age distribution")
p.on(ax).show()



# correlation plots
df_cor = raw_train \
    .select(cs.by_dtype(pl.Int64, pl.Float64)) \
    .select(pl.exclude('id','loan_status')) \
    .corr() \
    .with_columns(var2 = pl.Series(dfcor.columns)).unpivot(index = 'var2', value_name='correlation') \
    .filter(pl.col('variable') != pl.col('var2')) \
    .with_columns(pl.concat_str(['variable','var2'], separator = ',').str.split(',').list.eval(pl.element().sort()).alias('sorted_pair')) \
    .with_columns(rownum = pl.col('sorted_pair').cum_count().over('sorted_pair')) \
    .filter(pl.col('rownum') == 1) \
    .select(pl.exclude('sorted_pair','rownum')) \
    .with_columns(abs_corr = pl.col('correlation').abs()) \
    .sort(by = 'abs_corr', descending = True) \
    .select(pl.exclude('abs_corr'))

df_cor


# lets plot some inter-relationships
fig, ax = plt.subplots()
p = so.Plot(raw_train).add(so.Dots(), x = 'person_age', y = 'cb_person_cred_hist_length')
p.on(ax).show()

# person age & loan amount
fig, ax = plt.subplots()
p = so.Plot(raw_train).add(so.Dots(), x = 'person_age', y = 'loan_amnt')
p.on(ax).show()

# loan amnt and person income
fig, ax = plt.subplots()
p = so.Plot(raw_train.with_columns(loan_status = pl.col('loan_status').cast(pl.String()))).add(so.Dots(), x = 'person_income', y = 'loan_amnt', color = 'loan_status')
p.on(ax).show()
# seems like default is popular with lower income
fig, ax = plt.subplots()
p = so.Plot(raw_train.filter(pl.col('person_income') < 250000).with_columns(loan_status = pl.col('loan_status').cast(pl.String()))) \
    .add(so.Dots(), x = 'person_income', y = 'loan_amnt', color = 'loan_status')
p.on(ax).show()

# loan status by loan amnt
fig, ax = plt.subplots()
p = so.Plot(raw_train.with_columns(loan_status = pl.col('loan_status').cast(pl.String()))).add(so.Dots(), x = 'id', y = 'loan_amnt', color = 'loan_status')
p.on(ax).show()


# pair plots
# p = so.Plot(raw_train).pair(y = number_cols, x = number_cols).add(so.Dots())
# p.show()