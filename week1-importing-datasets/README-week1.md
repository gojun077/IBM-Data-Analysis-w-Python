Week 1: Importing Datasets
=============================

# Summary

- Created on: Sep 30 2023
- Created by: gojun077@gmail.com
- Last Updated: Sep 30 2023

# Topics

## Downloading open-source data sets for problem sets

The original link from the videos is outdated; the used car price
data set from *1985 Ward's Automotive Yearbook* created by Jeffrey
Schlimmer is now available from the following page:

https://archive.ics.uci.edu/dataset/10/automobile

This data set is hosted by the UC Irvine Machine Learning Repository. If
you click the *Download* button on the web page, `automobile.zip` will
be downloaded to your computer. After unzipping the file It should contain the
following:

- `misc`
- `imports-85.names`
- `imports-85.data`
- `Index`
- `app.css`

We don't need `Index` and `app.css`; `Index` simply contains a manifest of
files in the `.zip` archive while `app.css` is a Cascading Style Sheet
definition for displaying these files in a browser.

Note that the field names are not contained in `imports-85.data`, which is
in `CSV` format. Instead, you can find the 26 field names in
`imports-85.names`.


## Lab 1 - Importing Datasets - Used Cars Pricing (ungraded)

This lab uses Jupyter Notebook and launches in your browser. If you would
like to do the lab locally outside of Jupyter, you could just use `ipython`
from the terminal. The instructions below apply to doing the lab locally.

First you will need to install the following packages with `pip` Python
package installer:

```sh
pip3 install pandas matplotlib scipy seaborn tqdm
```

You can check if a Python package has been installed with `pip`
using the command `pip show <pkgName>`.

First let's open the CSV file named `imports-85.data`, assuming you
launched `ipython` while you are already in the same folder where you
downloaded the file:

```python
import pandas as pd
import numpy as np

df = pd.read_csv('imports-85.data', header=None)
```

To check if the csv was properly loaded into the dataframe, let's look
at the first 5 rows:

```python
df.head(5)
```

This should return:

```
   0    1            2    3    4     5            6    7   ...    18    19    20   21    22  23  24     25
0   3    ?  alfa-romero  gas  std   two  convertible  rwd  ...  3.47  2.68   9.0  111  5000  21  27  13495
1   3    ?  alfa-romero  gas  std   two  convertible  rwd  ...  3.47  2.68   9.0  111  5000  21  27  16500
2   1    ?  alfa-romero  gas  std   two    hatchback  rwd  ...  2.68  3.47   9.0  154  5000  19  26  16500
3   2  164         audi  gas  std  four        sedan  fwd  ...  3.19  3.40  10.0  102  5500  24  30  13950
4   2  164         audi  gas  std  four        sedan  4wd  ...  3.19  3.40   8.0  115  5500  18  22  17450

[5 rows x 26 columns]
```

To view the bottom 10 rows, you would use `df.tail(10)`.

Currently the fields are labeled from 0 to 25. The headers are not included
in the data file. We can find the field names from `import-85.names`,
and then manually add the names to a Python list.

```python
# create headers list
headers = ["symboling","normalized-losses","make","fuel-type",
"aspiration", "num-of-doors","body-style","drive-wheels",
"engine-location","wheel-base", "length","width","height",
"curb-weight","engine-type","num-of-cylinders", "engine-size",
"fuel-system","bore","stroke","compression-ratio","horsepower",
"peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)
```

Use the `.columns` function to assign the list above as field names
for our dataframe.

```python
df.columns = headers
df.head(10)
```

This should return the following:

```
   symboling normalized-losses         make fuel-type  ... peak-rpm city-mpg highway-mpg  price
0          3                 ?  alfa-romero       gas  ...     5000       21          27  13495
1          3                 ?  alfa-romero       gas  ...     5000       21          27  16500
2          1                 ?  alfa-romero       gas  ...     5000       19          26  16500
3          2               164         audi       gas  ...     5500       24          30  13950
4          2               164         audi       gas  ...     5500       18          22  17450
5          2                 ?         audi       gas  ...     5500       19          25  15250
6          1               158         audi       gas  ...     5500       19          25  17710
7          1                 ?         audi       gas  ...     5500       19          25  18920
8          1               158         audi       gas  ...     5500       17          20  23875
9          0                 ?         audi       gas  ...     5500       16          22      ?

[10 rows x 26 columns]
```

The `?` symbols above denote missing values. In `numpy` there is a *Not a
Number* type object denoted by `np.NaN`. Replace all occurrences of `?`
with `np.NaN` using `df.replace('?', np.NaN)`, and drop missing
values using the `.dropna()` method:

```python
df1 = df.replace('?', np.NaN)
df = df1.dropna(subset=["price"], axis=0)
```

Now if we look at `df.head(20)` we should see that all the rows missing
data for the field `price` (the last field) have been deleted (i.e. row
9 for the used *audi* above was deleted):

```
    symboling normalized-losses         make fuel-type  ... peak-rpm city-mpg highway-mpg  price
0           3               NaN  alfa-romero       gas  ...     5000       21          27  13495
1           3               NaN  alfa-romero       gas  ...     5000       21          27  16500
2           1               NaN  alfa-romero       gas  ...     5000       19          26  16500
3           2               164         audi       gas  ...     5500       24          30  13950
4           2               164         audi       gas  ...     5500       18          22  17450
5           2               NaN         audi       gas  ...     5500       19          25  15250
6           1               158         audi       gas  ...     5500       19          25  17710
7           1               NaN         audi       gas  ...     5500       19          25  18920
8           1               158         audi       gas  ...     5500       17          20  23875
10          2               192          bmw       gas  ...     5800       23          29  16430
11          0               192          bmw       gas  ...     5800       23          29  16925
12          0               188          bmw       gas  ...     4250       21          28  20970
13          0               188          bmw       gas  ...     4250       21          28  21105
14          1               NaN          bmw       gas  ...     4250       20          25  24565
15          0               NaN          bmw       gas  ...     5400       16          22  30760
16          0               NaN          bmw       gas  ...     5400       16          22  41315
17          0               NaN          bmw       gas  ...     5400       15          20  36880
18          2               121    chevrolet       gas  ...     5100       47          53   5151
19          1                98    chevrolet       gas  ...     5400       38          43   6295
20          0                81    chevrolet       gas  ...     5400       38          43   6575

[20 rows x 26 columns]
```

To find the name of columns in the dataframe, you can use the built-in
method `.columns`, which does not take any arguments. For example,
`df.columns` will return the following:

```
Index(['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration',
       'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',
       'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type',
       'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke',
       'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
       'highway-mpg', 'price'],
      dtype='object')
```

You can save your dataframe to a CSV file using `DataFrameName.to_csv()`.
Above we imported raw CSV data without any headers from `imports-85.data`
and added headers for the columns. We then removed rows lacking data in the
`price` field/column. To save the dataframe to a file you can use
`df.to_csv()`. It is also possible to write to other formats. Pandas
supports csv, json, excel, hdf, and sql OOTB.

| Data Format | Read              | Save            |
|-------------|-------------------|-----------------|
| csv         | `pd.read_json()`  | `df.to_csv()`   |
| json        | `pd.read_json()`  | `df.to_json()`  |
| excel       | `pd.read_excel()` | `df.to_excel()` |
| hdf         | `pd.read_hdf()`   | `df.to_hdf()`   |
| sql         | `pd.read_sql()`   | `df.to_sql()`   |

To examine the data types for data stored in a Pandas dataframe, use
`df.dtypes`. The supported types are `object`, `float`, `int`, `bool`, and
`datetime64`.

To get a statistical summary of each column, use `df.describe()`. For
our current dataframe, we get the following:

```
symboling  wheel-base      length       width  ...  engine-size  compression-ratio    city-mpg  highway-mpg
count  201.000000  201.000000  201.000000  201.000000  ...   201.000000         201.000000  201.000000   201.000000
mean     0.840796   98.797015  174.200995   65.889055  ...   126.875622          10.164279   25.179104    30.686567
std      1.254802    6.066366   12.322175    2.101471  ...    41.546834           4.004965    6.423220     6.815150
min     -2.000000   86.600000  141.100000   60.300000  ...    61.000000           7.000000   13.000000    16.000000
25%      0.000000   94.500000  166.800000   64.100000  ...    98.000000           8.600000   19.000000    25.000000
50%      1.000000   97.000000  173.200000   65.500000  ...   120.000000           9.000000   24.000000    30.000000
75%      2.000000  102.400000  183.500000   66.600000  ...   141.000000           9.400000   30.000000    34.000000
max      3.000000  120.900000  208.100000   72.000000  ...   326.000000          23.000000   49.000000    54.000000

[8 rows x 10 columns]
```

You can see that the summary stats are only shown for 10 cols, although we
have 26 cols total. By default, only stats for numeric type columns are
shown. If you want to see the stats for all columns, regardless of type,
use `df.describe(include = 'all')`

```
         symboling normalized-losses    make fuel-type  ... peak-rpm    city-mpg highway-mpg price
count   201.000000               164     201       201  ...      199  201.000000  201.000000   201
unique         NaN                51      22         2  ...       22         NaN         NaN   186
top            NaN               161  toyota       gas  ...     5500         NaN         NaN  8921
freq           NaN                11      32       181  ...       36         NaN         NaN     2
mean      0.840796               NaN     NaN       NaN  ...      NaN   25.179104   30.686567   NaN
std       1.254802               NaN     NaN       NaN  ...      NaN    6.423220    6.815150   NaN
min      -2.000000               NaN     NaN       NaN  ...      NaN   13.000000   16.000000   NaN
25%       0.000000               NaN     NaN       NaN  ...      NaN   19.000000   25.000000   NaN
50%       1.000000               NaN     NaN       NaN  ...      NaN   24.000000   30.000000   NaN
75%       2.000000               NaN     NaN       NaN  ...      NaN   30.000000   34.000000   NaN
max       3.000000               NaN     NaN       NaN  ...      NaN   49.000000   54.000000   NaN

[11 rows x 26 columns]
```

You can seelct specific columns in a dataframe by indicating the name of
each column, i.e., `dataframe[[' column 1 ',column 2', 'column 3']]`.
You can then apply dataframe methods on just the specified columns.

Let's get the summary stats of just the `length` and `compression-ratio`
fields:

```python
df[['length', 'compression-ratio']].describe()
```

This returns:

```
           length  compression-ratio
count  201.000000         201.000000
mean   174.200995          10.164279
std     12.322175           4.004965
min    141.100000           7.000000
25%    166.800000           8.600000
50%    173.200000           9.000000
75%    183.500000           9.400000
max    208.100000          23.000000
```

There is another built-in dataframe method `info()`, which provides
a summary of your DataFrame including index dtype, columns, non-null
values, and memory usage. Consider the output of `df.info()`:

```
<class 'pandas.core.frame.DataFrame'>
Index: 201 entries, 0 to 204
Data columns (total 26 columns):
 #   Column             Non-Null Count  Dtype
---  ------             --------------  -----
 0   symboling          201 non-null    int64
 1   normalized-losses  164 non-null    object
 2   make               201 non-null    object
 3   fuel-type          201 non-null    object
 4   aspiration         201 non-null    object
 5   num-of-doors       199 non-null    object
 6   body-style         201 non-null    object
 7   drive-wheels       201 non-null    object
 8   engine-location    201 non-null    object
 9   wheel-base         201 non-null    float64
 10  length             201 non-null    float64
 11  width              201 non-null    float64
 12  height             201 non-null    float64
 13  curb-weight        201 non-null    int64
 14  engine-type        201 non-null    object
 15  num-of-cylinders   201 non-null    object
 16  engine-size        201 non-null    int64
 17  fuel-system        201 non-null    object
 18  bore               197 non-null    object
 19  stroke             197 non-null    object
 20  compression-ratio  201 non-null    float64
 21  horsepower         199 non-null    object
 22  peak-rpm           199 non-null    object
 23  city-mpg           201 non-null    int64
 24  highway-mpg        201 non-null    int64
 25  price              201 non-null    object
dtypes: float64(5), int64(5), object(16)
memory usage: 42.4+ KB
```

