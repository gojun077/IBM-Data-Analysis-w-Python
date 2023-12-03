Week 1 Graded Quiz: Importing Data Sets
============================================

# Summary

- Created by: gopeterjun@naver.com
- Created on: Dec 3 2023 (日)
- Last Updated: Dec 3 2023 (日)

I previously completed this quiz at the end of September / beginning of
October 2023, but then I didn't continue with the course until December,
so all my progress was reset!

## Questions

**Note**: Toggle Markdown checklists in Emacs Markdown mode with `C-c C-d`

### 1. What Python library is primarily used for machine learning?

- [x] `scikit-learn`
- [ ] `numpy`
- [ ] `matplotlib`
- [ ] `pandas`

### 2. Given `headers_list = ['A','B','C']` and data frame `df`...

which has 3 columns, what syntax should you use to replace the headers of
the data frame `df` with values in the list `headers_list`?

- [ ] `df.tail(headers_list)`
- [x] `df.columns = headers_list`
- [ ] `df.tail() = headers_list`
- [ ] `df.head(headers_list)`

### 3. What task does `df = pandas.read_csv("A.csv")` perform?

- [ ] Changes the name of column in `df` to the ones in `"A.csv"`
- [ ] Displays the contents of the CSV file
- [ ] Saves the data frame `df` to a CSV file named `"A.csv"`
- [x] Loads the data from CSV file called `"A.csv"` into `df`

### 4. Consider the segment of the following data frame:


| | symboling | normalized-losses | make | fuel-type | aspiration | num-of-doors | body-style | drive-wheels | engine-location | wheel-base | ... | engine-size | fuel-system |
|-|-----------|-----------------|------|-----------|------------|-------------|---------|-------------|-----------------|-----------|-----|-------------|-------------|
|0|3|?|alfa-romero|gas|std|two|convertible|rwd|front|88.6|...|130|mpfi|
|1|3|?|alfa-remoro|gas|std|two|convertible|rwd|front|88.6|...|130|mpfi|
|2|2|164|audi|gas|std|four|sedan|fwd|front|99.8|...|109|109|mpfi|

What is the type of attribute *make*?

- [x] `object`
- [ ] `int64`
- [ ] `float64`
- [ ] `string`

**Note**: before `pandas`1.0, only `object` datatype was used to store
strings. Unfortunately, non-string data can also be stored using type
`object`. The above answer choice `string` is a trick; only in recent
versions of `pandas` does datatype `StringDtype` aka `string` exist!

`object` dtype is still the default for strings, but you can specify the
dtype as follows:

```python
# method 1
cities_string = pd.Series(['Houston','Rome','Madrid'], dtype=pd.StringDtype())
# method 2
cities_string = pd.Series(['Houston','Rome','Madrid'], dtype='string')
```

You can also convert from `object` to `string` dtype using `astype()`:

```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
print("Type of column B before converting: ", df['B'].dtype)
# above will print 'object'
df['B'] = df['B'].astype("string")
print("Type of column B after converting: ", df['B'].dtype)
# above will print 'string'
```

### 5. How do you generate descriptive statistics for all columns in `df`?

- [ ] `df.info`
- [ ] `df.statistics(includ = "all")`
- [x] `df.describe(include = "all")`
- [ ] `df.describe()`


