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
    - person_income


"""