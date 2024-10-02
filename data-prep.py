import polars as pl
import polars.selectors as cs

"""
PLAN
======

* reduce categories
    - person_home_ownership: rent or not
    - loan_grade: A, B, other
    
* center/scale data to remove skewness
    - as needed for logistic regression
    - not needed for tree-based models

* outliers
    - person_age; max out to 80
    - person_income; max out at 175K

"""

# improve print outputs from polars
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(3000)
pl.Config.set_tbl_cols(-1)

# load files
raw_train = pl.read_csv('data/train.csv')
raw_test = pl.read_csv('data/test.csv')

# reduce number of categories
def reduce_cats(df):
    ans = df \
        .with_columns(pl.when(pl.col('person_home_ownership') != 'RENT').then(pl.lit(0)).otherwise(pl.lit(1)).alias('renting')) \
        .with_columns(pl.when(pl.col('loan_grade') == 'A').then(pl.lit('a')).when(pl.col('loan_grade')=='B').then(pl.lit('b')).otherwise(pl.lit('other')).alias('new_loan_grade'))

    return(ans)

