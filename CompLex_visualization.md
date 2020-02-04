
# Визуализация данных

**План:**

1. pandas - организация данных
2. Matplotlib
3. Seaborn
4. Wordcloud


```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import random
random.seed = 23
```

**Обычный график**

Точки по х и у соединены линиями. Нужен, если есть однозначное соответствие х и у и мы хотим показать как при изменении х меняется у. Например, по х может быть время, а по у - частотность слова (как на графиках в НКРЯ).


```python
X = list(range(2010, 2020))
Y = [random.randint(i*10, (i+1)*20) for i in range(len(X))]
print('X:', X)
print('Y:', Y)

plt.plot(X, Y) # рисуем график - последовательно соединяем точки с координатами из X и Y
plt.title('Frequency of some random word') # заголовок
plt.ylabel('IPM') # подпись оси Х
plt.xlabel('Year') # подпись оси Y
plt.show()
```

    X: [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    Y: [4, 16, 48, 62, 83, 86, 68, 139, 155, 109]
    


![png](output_3_1.png)


**Scatter plot**

Точки, как и раньше, задаются по х и у, но теперь не соединяются линиями. Такие гарфики используются для отображения наблюдений в эксперименте, когда есть два параметра, которые могут принимать различные значения и нужно показать, какие комбинации есть и как они расположены.


```python
X = [1.2, 2, 3.1, 4, 5.1, 1.7, 1.5, 3.5, 4.5, 4, 2, 2, 2.7, 3.1, 4.1]
Y = [1, 1.4, 3, 4, 5, 1.5, 1.3, 3.5, 4.5, 4.5, 1.3, 2.1, 3.2, 5.1, 4.9]

plt.scatter(X, Y, color='purple', label='group 1') # меняем цвет, добавляем label
plt.scatter(Y, X, color='orange', label='group 2') # нарисуем еще какие-то значения на том же графике
plt.title('Combinations of two variables\n in our imaginary experiment') # перенос строки
plt.ylabel('Some other variable')
plt.xlabel('Some variable')
plt.legend(loc='best') # автоматический поиск места для легенды
plt.show()
```


![png](output_5_0.png)



```python
X = [1.2, 2, 3.1, 4, 5.1, 1.7, 1.5, 3.5, 4.5, 4, 2, 2, 2.7, 3.1, 4.1]
Y = [1, 1.4, 3, 4, 5, 1.5, 1.3, 3.5, 4.5, 4.5, 1.3, 2.1, 3.2, 5.1, 4.9]
labels = 'abcdefghijklmno'
plt.figure(figsize=(6, 6)) # размер графика
plt.scatter(X, Y, color='red', label='group 1', marker='^') # маркер - треугольник
for x, y, key in zip(X, Y, range(len(X))):
    plt.text(x+0.2, y, labels[key]) # подписи
plt.title('Combinations of two variables\n in our imaginary experiment')
plt.ylabel('Some other variable')
plt.xlabel('Some variable')
plt.xlim((0, 7)) # предел по х
plt.ylim((0, 7)) # предел по у
plt.show()
```


![png](output_6_0.png)


**Bar plot**

Столбчатая диграмма - для категориальных данных по х и чисел по у, например, если у нас есть дни недели и среднее количество ругательств, которое человек произносит в этот день.


```python
X = [1, 2, 3, 4, 5]
X2 = [6, 7] # сделаем выходные отдельно
Y = [30, 15, 17, 15, 10]
Y2 = [7, 3]
DAYS = ['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс']
plt.bar(X, Y, color='grey')
plt.bar(X2, Y2, color='pink')
plt.xticks(ticks=X+X2, labels=DAYS)
plt.title('Среднее количество ругательств по дням недели')
plt.ylabel('Среднее кол-во ругательств')
plt.xlabel('День недели')
plt.show()
```


![png](output_8_0.png)


**Dendrogram**

https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/


```python
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
```


```python
%matplotlib inline
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
```


```python
# generate two clusters: a with 100 points, b with 50:
np.random.seed(4711)
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
X = np.concatenate((a, b),)
print(X.shape)  # 150 samples with 2 dimensions
plt.scatter(X[:,0], X[:,1])
plt.show()
```

    (150, 2)
    


![png](output_12_1.png)



```python
# generate the linkage matrix
Z = linkage(X, 'ward')#the Ward variance minimization algorithm
```


```python
# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
```


![png](output_14_0.png)


**Heatmap**

Хитмэп нужен, когда у нас есть 3 переменные.


```python
X = [
    [1, 2, 3, 4, 5],
    [2, 3, 4, 5, 6],
    [3, 4, 5, 6, 7],
    [4, 5, 6, 7, 8],
    [5, 6, 7, 8, 9],
]
sns.heatmap(
    X, # матрица значений
    annot=True, # значения из матрицы
    xticklabels=DAYS[:5],
    yticklabels=[f'степень {i}' for i in range(0, 5)]
)
plt.xlabel('День недели')
plt.ylabel('Степень выраженности ругания')
plt.ylabel('Количесто ругающихся людей\n по дню недели и степени выраженности ругания')
plt.show()
```


![png](output_16_0.png)


## Pandas


```python
import pandas as pd
```

**Создать датафрейм (таблицу)**

Можно задать целые колонки


```python
dictionary = {'name': ['Джон', 'Мария', 'Алекс'], 'age': [21, 29, 35]}

df = pd.DataFrame(dictionary)

df
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
      <th>name</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Джон</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Мария</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Алекс</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>



Можно описать по рядам. Это чем-то напоминает майстем, правда?


```python
name_list = [{'name': 'Джон', 'age': 21}, {'name': 'Мария', 'age': 29}, {'name': 'Алекс', 'age': 35}]

df = pd.DataFrame(name_list)

df
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
      <th>age</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21</td>
      <td>Джон</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29</td>
      <td>Мария</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>Алекс</td>
    </tr>
  </tbody>
</table>
</div>




```python
n_l = [{'text': 'Но', 'analysis': [{'wt': 0.9998906255000001, 'lex': 'но', 'gr': 'CONJ='}]}, {'text': ' '}, {'text': 'не', 'analysis': [{'wt': 1, 'lex': 'не', 'gr': 'PART='}]}, {'text': ' '}, {'text': 'становится', 'analysis': [{'wt': 1, 'lex': 'становиться', 'gr': 'V,нп=непрош,ед,изъяв,3-л,несов'}]}]
print(n_l)
```

    [{'text': 'Но', 'analysis': [{'wt': 0.9998906255000001, 'lex': 'но', 'gr': 'CONJ='}]}, {'text': ' '}, {'text': 'не', 'analysis': [{'wt': 1, 'lex': 'не', 'gr': 'PART='}]}, {'text': ' '}, {'text': 'становится', 'analysis': [{'wt': 1, 'lex': 'становиться', 'gr': 'V,нп=непрош,ед,изъяв,3-л,несов'}]}]
    

Добавить столбцы


```python
for word in n_l:
    if 'analysis' in word:
        gr = word['analysis'][0]['gr']
        pos = gr.split('=')[0].split(',')[0]
        w= 'word'
        p = 'pos'
        d = [{w:word['text'],p:pos}]
        print(d)
```

    [{'word': 'Но', 'pos': 'CONJ'}]
    [{'word': 'не', 'pos': 'PART'}]
    [{'word': 'становится', 'pos': 'V'}]
    


```python
df['country'] = ['USA', 'Spain', 'UK']

df
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
      <th>name</th>
      <th>age</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Джон</td>
      <td>21</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Мария</td>
      <td>29</td>
      <td>Spain</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Алекс</td>
      <td>35</td>
      <td>UK</td>
    </tr>
  </tbody>
</table>
</div>



Удалить столбцы


```python
del df ['country']

df
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
      <th>name</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Джон</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Мария</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Алекс</td>
      <td>35</td>
    </tr>
  </tbody>
</table>
</div>



Применить какую-то функцию. Функция принимает один аргумент и отдает один объект


```python
def get_age_group(age):
    if age > 30:
        return '30+'
    else:
        return '30-'
```


```python
df['age_group'] = df['age'].apply(get_age_group)

df
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
      <th>name</th>
      <th>age</th>
      <th>age_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Джон</td>
      <td>21</td>
      <td>30-</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Мария</td>
      <td>29</td>
      <td>30-</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Алекс</td>
      <td>35</td>
      <td>30+</td>
    </tr>
  </tbody>
</table>
</div>



**Сохранить в файл**


```python
df.to_csv(
    'some_file.csv',  # имя файла
    sep='\t',  # разделитель - лучше использовать таб
    index=False,  # не сохранять номера строк (слева)
)
```

Или в одну строчку


```python
df.to_csv('some_file.csv', sep='\t', index=False)
```

**Прочитать из csv-файла**


```python
tolstoy = pd.read_csv('tolstoy.csv', sep='\t').fillna('') # читаем файл и пустые значения заполняем пустыми строками
```


```python
tolstoy.head(10)
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
      <th>lex</th>
      <th>word</th>
      <th>POS</th>
      <th>time</th>
      <th>gender</th>
      <th>case</th>
      <th>number</th>
      <th>verbal</th>
      <th>adj_form</th>
      <th>comp</th>
      <th>...</th>
      <th>имя</th>
      <th>отч</th>
      <th>фам</th>
      <th>вводн</th>
      <th>гео</th>
      <th>сокр</th>
      <th>обсц</th>
      <th>разг</th>
      <th>редк</th>
      <th>устар</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>том</td>
      <td>том</td>
      <td>S</td>
      <td></td>
      <td>муж</td>
      <td>вин</td>
      <td>ед</td>
      <td></td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>первый</td>
      <td>первый</td>
      <td>ANUM</td>
      <td></td>
      <td>муж</td>
      <td>вин</td>
      <td>ед</td>
      <td></td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>часть</td>
      <td>часть</td>
      <td>S</td>
      <td></td>
      <td>жен</td>
      <td>вин</td>
      <td>ед</td>
      <td></td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>первый</td>
      <td>первая</td>
      <td>ANUM</td>
      <td></td>
      <td>жен</td>
      <td>им</td>
      <td>ед</td>
      <td></td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>ну</td>
      <td>ну</td>
      <td>PART</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td>здравствовать</td>
      <td>здравствуйте</td>
      <td>V</td>
      <td></td>
      <td></td>
      <td></td>
      <td>мн</td>
      <td>пов</td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td>здравствовать</td>
      <td>здравствуйте</td>
      <td>V</td>
      <td></td>
      <td></td>
      <td></td>
      <td>мн</td>
      <td>пов</td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>садиться</td>
      <td>садитесь</td>
      <td>V</td>
      <td>непрош</td>
      <td></td>
      <td></td>
      <td>мн</td>
      <td>изъяв</td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td>и</td>
      <td>и</td>
      <td>CONJ</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td>рассказывать</td>
      <td>рассказывайте</td>
      <td>V</td>
      <td></td>
      <td></td>
      <td></td>
      <td>мн</td>
      <td>пов</td>
      <td></td>
      <td></td>
      <td>...</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
<p>10 rows × 25 columns</p>
</div>




```python
tolstoy['gender'].value_counts()
```




            107839
    муж      56781
    жен      33320
    сред     18063
    мж          96
    Name: gender, dtype: int64




```python
tolstoy[tolstoy['gender'] == 'мж']['lex'].value_counts().head(10)
```




    лаврушка       17
    браунау        11
    душенька        6
    бедняжка        5
    моро            5
    судья           5
    плакса          3
    староста        3
    сорвиголова     2
    бондаренко      2
    Name: lex, dtype: int64




```python
tolstoy['gender'].value_counts().plot.bar(color='purple'); # барплот сразу из датафрейма
plt.title('Gender')
plt.xlabel('gender')
plt.ylabel('number of entries');
```


![png](output_42_0.png)



```python
df2 = tolstoy[
    (tolstoy['gender'] != '') & (tolstoy['gender'] != 'мж')
][
    ['POS', 'gender', 'number']
].groupby(['POS', 'gender'], as_index=False).count()

df2.columns = ['POS', 'gender', 'total']
sns.barplot(x="POS", y="total", hue='gender', data=df2)
plt.title('Gender by POS')
plt.xlabel('POS')
plt.ylabel('number of entries');
```


![png](output_43_0.png)


**Pie chart**

Таким образом удобно визуализировать доли, которые занимают категории внтури целого, однако нужно аккуратно использовать этот вид графиков, потому что они могут ввести в заблуждение при небольшом количестве данных или при сравнении групп.


```python
plt.figure(figsize=(6, 6))
tolstoy['gender'].value_counts().plot(kind='pie');
plt.title('Gender');
```


![png](output_45_0.png)



```python
df2 = tolstoy[['lex', 'POS', 'gender']].groupby(['lex', 'POS'], as_index=False).count()
df2.columns = ['lex', 'POS', 'total']
df2 = df2[df2['total'] > 10]
```

**Box plot**

Боксплоты нужны для того, чтобы показать различия в распределениях в разных группах, например, если у нас есть части речи и разные частоты лемм этих частей речи.


```python
plt.figure(figsize=(10, 6))
sns.boxplot(x="POS", y="total", data=df2)
plt.ylim((0, 1500))
plt.title('N of lemma entries by POS')
plt.ylabel('n entries')
plt.xlabel('POS');
```


![png](output_48_0.png)


**Гистограмма**

Главное отличие гистограммы от барплота - на гистограмме у нас одна переменная и мы хотим изучить только ее: сколько объектов с тем или иным значением (в промежуке значений), а барплот - это значения по категориям.


```python
df2['length'] = df2['lex'].apply(len)
plt.figure(figsize=(10, 6))
sns.distplot(df2['length'], bins=17, color='green')
plt.title('Distribution of lemma length')
plt.ylabel('%')
plt.xlabel('Length of word');
```


![png](output_50_0.png)


### Wordcloud

Один из видов визуализации текста - это облако слов. В зависимости от частотности слова меняется его размер на картинке


```python
from wordcloud import WordCloud
from nltk.corpus import stopwords

stops = set(stopwords.words('russian') + ['это', 'весь', 'который', 'мочь', 'свой'])
```


```python
text = ' '.join(tolstoy['lex'])

wordcloud = WordCloud(
    background_color ='white',
    width = 800,
    height = 800, 
).generate(text)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud)
plt.axis("off") 
plt.title('Облако слов (включая стоп-слова)')
plt.show()
```


![png](output_53_0.png)



```python
text = ' '.join([word for word in tolstoy['lex'].values if word not in stops])
```


```python
wordcloud = WordCloud(
    background_color ='white',
    width = 800,
    height = 800, 
).generate(text)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud)
plt.axis("off") 
plt.title('Облако слов (без сто-слов)')
plt.show()
```


![png](output_55_0.png)

