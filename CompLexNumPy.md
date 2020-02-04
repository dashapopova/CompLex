
### NumPy

Зачем нам нужен NumPy?

+ позволяет проводить кучу операций с векторами и матрицами
+ быстрее, чем for-loop имплементация
+ используется в машинном обучении и AI
+ массивы NumPy используются в других модулях питона

В Jupyter notebooks, NumPy документация: Help -> NumPy reference.


```python
import numpy as np
```

### Vectors

#### Vector Initialization


```python
np.zeros(5)
```




    array([0., 0., 0., 0., 0.])




```python
np.ones(5)
```




    array([1., 1., 1., 1., 1.])




```python
# convert list to numpy array
np.array([1,2,3,4,5])
```




    array([1, 2, 3, 4, 5])




```python
# convert numpy array to list
np.ones(5).tolist()
```




    [1.0, 1.0, 1.0, 1.0, 1.0]




```python
# one float => all floats
np.array([1.0,2,3,4,5])
```




    array([1., 2., 3., 4., 5.])




```python
# same as above
np.array([1,2,3,4,5], dtype='float')
```




    array([1., 2., 3., 4., 5.])




```python
# spaced values in interval
np.array([x for x in range(20) if x % 2 == 0])
```




    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])




```python
# same as above
np.arange(0,20,2)
```




    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])




```python
# random floats in [0, 1)
np.random.random(10)
```




    array([0.65276471, 0.30669114, 0.28685996, 0.16553095, 0.66394034,
           0.63536315, 0.76223511, 0.99754089, 0.19131857, 0.75089894])




```python
# random integers
np.random.randint(5, 15, size=10)
```




    array([10,  9, 12,  9,  6,  5, 10, 13, 13,  6])



#### Vector indexing


```python
x = np.array([10,20,30,40,50])
```


```python
x[0]
```




    10




```python
# slice
x[0:2]
```




    array([10, 20])




```python
x[0:1000]
```




    array([10, 20, 30, 40, 50])




```python
# last value
x[-1]
```




    50




```python
# last value as array
x[[-1]]
```




    array([50])




```python
# last 3 values
x[-3:]
```




    array([30, 40, 50])




```python
# pick indices
x[[0,2,4]]
```




    array([10, 30, 50])



#### Vector assignment


```python
#x2 = x # try this line instead
x2 = x.copy()
```


```python
x2[0] = 10
```


```python
x2
```




    array([10, 20, 30, 40, 50])




```python
x2[[1,2]] = 10
```


```python
x2
```




    array([10, 10, 10, 40, 50])




```python
x2[[3,4]] = [0, 1]

x2
```




    array([10, 10, 10,  0,  1])




```python
# check if the original vector changed
x
```




    array([10, 20, 30, 40, 50])



#### Vectorized operations


```python
x.sum()
```




    150




```python
x.mean()
```




    30.0




```python
x.max()
```




    50




```python
x.argmax()
```




    4




```python
np.log(x)
```




    array([2.30258509, 2.99573227, 3.40119738, 3.68887945, 3.91202301])




```python
np.exp(x)
```




    array([2.20264658e+04, 4.85165195e+08, 1.06864746e+13, 2.35385267e+17,
           5.18470553e+21])




```python
x + x  # Try also with *, -, /, etc.
```




    array([ 20,  40,  60,  80, 100])




```python
x + 1
```




    array([11, 21, 31, 41, 51])



#### Сравнение с питоновскими списками

Векторизация математических выражений очень выигрышна. Давайте проверим! Применим np.log к каждому элементу списка с 10 млн. значений, проведём ту же операцию с вектором.



```python
# log every value as list, one by one
def listlog(vals):
    return [np.log(y) for y in vals]
```


```python
# get random vector
samp = np.random.random_sample(int(1e7))+1
samp
```




    array([1.65173729, 1.32995523, 1.65229234, ..., 1.84420545, 1.43324933,
           1.02993971])




```python
%time _ = np.log(samp)
```

    Wall time: 233 ms
    


```python
%time _ = listlog(samp)
```

    Wall time: 39.7 s
    

### Матрицы

Используются при машинном обучении

#### Matrix initialization


```python
np.array([[1,2,3], [4,5,6]])
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
np.array([[1,2,3], [4,5,6]], dtype='float')
```




    array([[1., 2., 3.],
           [4., 5., 6.]])




```python
np.zeros((3,5))
```




    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])




```python
np.ones((3,5))
```




    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]])




```python
np.identity(3)
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])




```python
np.diag([1,2,3])
```




    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]])



#### Matrix indexing


```python
X = np.array([[1,2,3], [4,5,6]])
X
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
X[0]
```




    array([1, 2, 3])




```python
X[0,0]
```




    1




```python
# get row
X[0, : ]
```




    array([1, 2, 3])




```python
# get column
X[ : , 0]
```




    array([1, 4])




```python
# get multiple columns
X[ : , [0,2]]
```




    array([[1, 3],
           [4, 6]])



#### Matrix assignment


```python
# X2 = X # try this line instead
X2 = X.copy()
X2
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
X2[0,0] = 20

X2

```




    array([[20,  2,  3],
           [ 4,  5,  6]])




```python
X2[0] = 3

X2

```




    array([[3, 3, 3],
           [4, 5, 6]])




```python
X2[: , -1] = [5, 6]

X2
```




    array([[3, 3, 5],
           [4, 5, 6]])




```python
# check if original matrix changed
X
```




    array([[1, 2, 3],
           [4, 5, 6]])



#### Matrix reshaping


```python
z = np.arange(1, 7)

z

```




    array([1, 2, 3, 4, 5, 6])




```python
z.shape

```




    (6,)




```python
Z = z.reshape(2,3)

Z
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
Z.shape

```




    (2, 3)




```python
Z.reshape(6)

```




    array([1, 2, 3, 4, 5, 6])




```python
# same as above
Z.flatten()
```




    array([1, 2, 3, 4, 5, 6])




```python
# transpose
Z.T
```




    array([[1, 4],
           [2, 5],
           [3, 6]])



#### Numeric operations


```python
A = np.array(range(1,7), dtype='float').reshape(2,3)

A
```




    array([[1., 2., 3.],
           [4., 5., 6.]])




```python
B = np.array([1, 2, 3])

```


```python
# not the same as A.dot(B)
A * B

```




    array([[ 1.,  4.,  9.],
           [ 4., 10., 18.]])




```python
A + B
```




    array([[2., 4., 6.],
           [5., 7., 9.]])




```python
A / B
```




    array([[1. , 1. , 1. ],
           [4. , 2.5, 2. ]])




```python
# matrix multiplication
A.dot(B)

```




    array([14., 32.])




```python
B.dot(A.T)

```




    array([14., 32.])




```python
A.dot(A.T)

```




    array([[14., 32.],
           [32., 77.]])




```python
# outer product
# multiplying each element of first vector by each element of the second
np.outer(B, B)


```




    array([[1, 2, 3],
           [2, 4, 6],
           [3, 6, 9]])



Пример применения такихх операций в нейронных сетях: https://cs231n.github.io/neural-networks-case-study/

### Применение NumPy с другими пакетами питона

#### Pandas

Можно привратить матрицы numpy в дейтафреймы pandas. Например, тогда можно присвоить названия строкам.


```python
import pandas as pd
```


```python
count_df = pd.DataFrame(
    np.array([
        [1,0,1,0,0,0],
        [0,1,0,1,0,0],
        [1,1,1,1,0,0],
        [0,0,0,0,1,1],
        [0,0,0,0,0,1]], dtype='float64'),
    index=['gnarly', 'wicked', 'awesome', 'lame', 'terrible'])
count_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gnarly</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>wicked</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>awesome</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>lame</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>terrible</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### SciPy

SciPy содержит множество операций линейной алгебры, оптимизации и т.п. Все операции могут работать с массивами NumPy.



```python
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy import linalg
```


```python
# cosine distance
a = np.random.random(10)
b = np.random.random(10)
cosine(a, b)
```




    0.17829535737532554




```python
# pearson correlation (coeff, p-value)
pearsonr(a, b)
```




    (0.1973323719176591, 0.584755396534702)




```python
# inverse of matrix
A = np.array([[1,3,5],[2,5,1],[2,3,8]])
linalg.inv(A)
```




    array([[-1.48,  0.36,  0.88],
           [ 0.56,  0.08, -0.36],
           [ 0.16, -0.12,  0.04]])



#### Matplotlib


```python
import matplotlib.pyplot as plt
```


```python
a = np.sort(np.random.random(30))
b = a**2
c = np.log(a)
plt.plot(a, b, label='y = x^2')
plt.plot(a, c, label='y = log(x)')
plt.legend()
plt.title("Some functions")
plt.show()
```


![png](output_93_0.png)

