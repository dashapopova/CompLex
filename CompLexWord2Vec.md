
# Векторные модели. Word2Vec

## Кратко о существующих системах

**GloVe**

GloVe берет и строит полную матрицу совместной встречаемости и после этого с помощью алгоритмов уменьшения размерности преобразует ее так, чтобы вектора были опредленной длины


**Word2Vec**

Это уже нейросеть и она на основе корпуса постепенно подбирает коэффициенты (значения в векторах) для каждого слова так, чтобы с помощью них можно было наилучшим образом предсказывать слова по контексту

**FastText**

Если мы берем конкретные слова, мы не можем ничего сказать о тех, что нам не встретились (например, уже видели вагон и строитель, а вот вагоностроителя у нас не было). Если мы возьмем слова не целиком, а в виде будквенных нграмм, то мы сможем сложить неизвестные слова.

**AdaGram**

Все предыдущие модели основаны на графических оболочках и не учитывают многозначность и омонимию. Есть только один вектор для слова "ключ" и мы ничего с этим не можем сделать. AdaGram исходит из предположения, что у слова есть n вариантов и если они действительно отличаются и достаточно часто встречаются, он умеет их разделить.

**BERT и ELMo**

Эти модели не просто могут отличить значения слов, о и скорректировать их вектора в зависимости от контекста, например, понять, что в отрывках “чистый ключ в лесной чаще” и “ключ от квартиры” совсем разные “ключи”. 

### Word2Vec

Одной из самых известных моделей для работы с дистрибутивной семантикой является word2vec. Технология основана на нейронной сети, предсказывающей вероятность встретить слово в заданном контексте. Этот инструмент был разработан группой исследователей Google в 2013 году, руководителем проекта был Томаш Миколов (сейчас работает в Facebook). Вот две самые главные статьи:

+ [Efficient Estimation of Word Representations inVector Space](https://arxiv.org/pdf/1301.3781.pdf)
+ [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)

Полученные таким образом вектора называются распределенными представлениями слов, или **эмбеддингами**.

#### Как это обучается?

Мы задаём вектор для каждого слова с помощью матрицы $w$ и вектор контекста с помощью матрицы $W$. По сути, word2vec является обобщающим названием для двух архитектур Skip-Gram и Continuous Bag-Of-Words (CBOW).

+ **CBOW** предсказывает текущее слово, исходя из окружающего его контекста.

+ **Skip-gram**, наоборот, использует текущее слово, чтобы предугадывать окружающие его слова.

#### Как это работает?

Word2vec принимает большой текстовый корпус в качестве входных данных и сопоставляет каждому слову вектор, выдавая координаты слов на выходе. Сначала он создает словарь, «обучаясь» на входных текстовых данных, а затем вычисляет векторное представление слов. Векторное представление основывается на контекстной близости: слова, встречающиеся в тексте рядом с одинаковыми словами (а следовательно, согласно дистрибутивной гипотезе, имеющие схожий смысл), в векторном представлении будут иметь близкие координаты векторов-слов. Для вычисления близости слов используется косинусное расстояние между их векторами.

С помощью дистрибутивных векторных моделей можно строить семантические пропорции (они же аналогии) и решать примеры:

+ король: мужчина = королева: женщина $\Rightarrow$
+ король - мужчина + женщина = королева

![w2v](https://cdn-images-1.medium.com/max/2600/1*sXNXYfAqfLUeiDXPCo130w.png)

Ещё про механику с картинками [тут](https://habr.com/ru/post/446530/)

#### Зачем это нужно?

+ используется для решения семантических задач
+ давайте подумаем, для описания каких семантических классов слов дистрибутивная информация особенно важна?
+ несколько интересных статей по дистрибутивной семантике:

* [Turney and Pantel 2010](https://jair.org/index.php/jair/article/view/10640)
* [Lenci 2018](https://www.annualreviews.org/doi/abs/10.1146/annurev-linguistics-030514-125254?journalCode=linguistics)
* [Smith 2019](https://arxiv.org/pdf/1902.06006.pdf)
* [Pennington et al. 2014](https://www.aclweb.org/anthology/D14-1162/)
* [Faruqui et al. 2015](https://www.aclweb.org/anthology/N15-1184/)

+ подаётся на вход нейронным сетям
+ используется в Siri, Google Assistant, Alexa, Google Translate...

#### Gensim

Использовать предобученную модель эмбеддингов или обучить свою можно с помощью библиотеки `gensim`. Вот ее [документация](https://radimrehurek.com/gensim/models/word2vec.html). Вообще-то `gensim` — библиотека для тематического моделирования текстов, но один из компонентов в ней — реализация на python алгоритмов из библиотеки word2vec (которая в оригинале была написана на C++).

Если gensim у вас не стоит, то ставим: `pip install gensim`. Можно сделать это прямо из jupyter'а! Чтобы выполнить какую-то команду не в питоне, в командной строке, нужно написать перед ней восклицательный знак.



```python
import re
import gensim
import logging
import nltk.data
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec

import warnings
warnings.filterwarnings('ignore')
```

    C:\ProgramData\Anaconda3\lib\site-packages\smart_open\ssh.py:34: UserWarning: paramiko missing, opening SSH/SCP/SFTP paths will be disabled.  `pip install paramiko` to suppress
      warnings.warn('paramiko missing, opening SSH/SCP/SFTP paths will be disabled.  `pip install paramiko` to suppress')
    

#### Как обучить свою модель

NB! Обратите внимание, что тренировка модели не включает препроцессинг! Это значит, что избавляться от пунктуации, приводить слова к нижнему регистру, лемматизировать их, проставлять частеречные теги придется до тренировки модели (если, конечно, это необходимо для вашей задачи). Т.е. в каком виде слова будут в исходном тексте, в таком они будут и в модели.

Поскольку иногда тренировка модели занимает много времени, то можно ещё вести лог событий, чтобы понимать, что на каком этапе происходит.


```python
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```

На вход модели даем текстовый файл, каждое предложение на отдельной строчке. Вот игрушечный пример с текстом «Бедной Лизы». Он заранее очищен от пунктуации, приведен к нижнему регистру и лемматизирован.


```python
f = 'liza_lem.txt'
data = gensim.models.word2vec.LineSentence(f)
```

Инициализируем модель. Основные параметры:

+ данные должны быть итерируемым объектом
+ size — размер вектора,
+ window — размер окна наблюдения,
+ min_count — мин. частотность слова в корпусе,
+ sg — используемый алгоритм обучения (0 — CBOW, 1 — Skip-gram),
+ sample — порог для downsampling'a высокочастотных слов,
+ workers — количество потоков,
+ alpha — learning rate,
+ iter — количество итераций,
+ max_vocab_size — позволяет выставить ограничение по памяти при создании словаря (т.е. если ограничение привышается, то низкочастотные слова будут выбрасываться). Для сравнения: 10 млн слов = 1Гб RAM.


```python
%time model_liza = gensim.models.Word2Vec(data, size=300, window=5, min_count=2, iter=20)
```

    2019-10-17 20:34:43,388 : INFO : collecting all words and their counts
    2019-10-17 20:34:43,396 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:43,407 : INFO : PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
    2019-10-17 20:34:43,415 : INFO : collected 1213 word types from a corpus of 3109 raw words and 392 sentences
    2019-10-17 20:34:43,420 : INFO : Loading a fresh vocabulary
    2019-10-17 20:34:43,427 : INFO : effective_min_count=2 retains 478 unique words (39% of original 1213, drops 735)
    2019-10-17 20:34:43,431 : INFO : effective_min_count=2 leaves 2374 word corpus (76% of original 3109, drops 735)
    2019-10-17 20:34:43,439 : INFO : deleting the raw counts dictionary of 1213 items
    2019-10-17 20:34:43,441 : INFO : sample=0.001 downsamples 83 most-common words
    2019-10-17 20:34:43,445 : INFO : downsampling leaves estimated 1817 word corpus (76.6% of prior 2374)
    2019-10-17 20:34:43,450 : INFO : estimated required memory for 478 words and 300 dimensions: 1386200 bytes
    2019-10-17 20:34:43,458 : INFO : resetting layer weights
    2019-10-17 20:34:43,482 : INFO : training model with 3 workers on 478 vocabulary and 300 features, using sg=0 hs=0 sample=0.001 negative=5 window=5
    2019-10-17 20:34:43,496 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:43,528 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:43,534 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:44,419 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:44,424 : INFO : EPOCH - 1 : training on 3109 raw words (1817 effective words) took 0.9s, 2001 effective words/s
    2019-10-17 20:34:44,434 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:44,479 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:44,484 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:45,508 : INFO : EPOCH 2 - PROGRESS: at 100.00% examples, 1711 words/s, in_qsize 0, out_qsize 1
    2019-10-17 20:34:45,513 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:45,516 : INFO : EPOCH - 2 : training on 3109 raw words (1795 effective words) took 1.1s, 1698 effective words/s
    2019-10-17 20:34:45,544 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:45,578 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:45,583 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:46,511 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:46,517 : INFO : EPOCH - 3 : training on 3109 raw words (1820 effective words) took 1.0s, 1915 effective words/s
    2019-10-17 20:34:46,529 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:46,567 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:46,571 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:47,675 : INFO : EPOCH 4 - PROGRESS: at 100.00% examples, 1628 words/s, in_qsize 0, out_qsize 1
    2019-10-17 20:34:47,680 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:47,682 : INFO : EPOCH - 4 : training on 3109 raw words (1822 effective words) took 1.1s, 1618 effective words/s
    2019-10-17 20:34:47,694 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:47,732 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:47,735 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:48,693 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:48,699 : INFO : EPOCH - 5 : training on 3109 raw words (1808 effective words) took 1.0s, 1847 effective words/s
    2019-10-17 20:34:48,709 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:48,758 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:48,762 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:49,661 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:49,663 : INFO : EPOCH - 6 : training on 3109 raw words (1814 effective words) took 0.9s, 1955 effective words/s
    2019-10-17 20:34:49,677 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:49,725 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:49,728 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:50,658 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:50,664 : INFO : EPOCH - 7 : training on 3109 raw words (1819 effective words) took 1.0s, 1914 effective words/s
    2019-10-17 20:34:50,676 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:50,722 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:50,740 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:51,719 : INFO : EPOCH 8 - PROGRESS: at 100.00% examples, 1775 words/s, in_qsize 0, out_qsize 1
    2019-10-17 20:34:51,724 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:51,728 : INFO : EPOCH - 8 : training on 3109 raw words (1798 effective words) took 1.0s, 1760 effective words/s
    2019-10-17 20:34:51,754 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:51,787 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:51,789 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:52,697 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:52,700 : INFO : EPOCH - 9 : training on 3109 raw words (1826 effective words) took 0.9s, 1974 effective words/s
    2019-10-17 20:34:52,707 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:52,752 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:52,756 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:53,698 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:53,703 : INFO : EPOCH - 10 : training on 3109 raw words (1832 effective words) took 1.0s, 1921 effective words/s
    2019-10-17 20:34:53,715 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:53,761 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:53,770 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:54,657 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:54,664 : INFO : EPOCH - 11 : training on 3109 raw words (1808 effective words) took 0.9s, 1960 effective words/s
    2019-10-17 20:34:54,675 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:54,719 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:54,722 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:55,701 : INFO : EPOCH 12 - PROGRESS: at 100.00% examples, 1777 words/s, in_qsize 0, out_qsize 1
    2019-10-17 20:34:55,706 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:55,711 : INFO : EPOCH - 12 : training on 3109 raw words (1778 effective words) took 1.0s, 1760 effective words/s
    2019-10-17 20:34:55,728 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:55,769 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:55,771 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:56,946 : INFO : EPOCH 13 - PROGRESS: at 100.00% examples, 1557 words/s, in_qsize 0, out_qsize 1
    2019-10-17 20:34:56,950 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:56,952 : INFO : EPOCH - 13 : training on 3109 raw words (1833 effective words) took 1.2s, 1549 effective words/s
    2019-10-17 20:34:56,958 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:56,995 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:57,002 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:58,142 : INFO : EPOCH 14 - PROGRESS: at 100.00% examples, 1593 words/s, in_qsize 0, out_qsize 1
    2019-10-17 20:34:58,146 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:58,150 : INFO : EPOCH - 14 : training on 3109 raw words (1836 effective words) took 1.2s, 1581 effective words/s
    2019-10-17 20:34:58,159 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:58,192 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:58,203 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:34:59,313 : INFO : EPOCH 15 - PROGRESS: at 100.00% examples, 1600 words/s, in_qsize 0, out_qsize 1
    2019-10-17 20:34:59,315 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:34:59,317 : INFO : EPOCH - 15 : training on 3109 raw words (1815 effective words) took 1.1s, 1594 effective words/s
    2019-10-17 20:34:59,323 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:34:59,355 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:34:59,357 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:35:00,412 : INFO : EPOCH 16 - PROGRESS: at 100.00% examples, 1680 words/s, in_qsize 0, out_qsize 1
    2019-10-17 20:35:00,415 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:35:00,419 : INFO : EPOCH - 16 : training on 3109 raw words (1797 effective words) took 1.1s, 1669 effective words/s
    2019-10-17 20:35:00,428 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:35:00,462 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:35:00,466 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:35:01,390 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:35:01,395 : INFO : EPOCH - 17 : training on 3109 raw words (1822 effective words) took 0.9s, 1932 effective words/s
    2019-10-17 20:35:01,405 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:35:01,454 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:35:01,461 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:35:02,553 : INFO : EPOCH 18 - PROGRESS: at 100.00% examples, 1637 words/s, in_qsize 0, out_qsize 1
    2019-10-17 20:35:02,560 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:35:02,563 : INFO : EPOCH - 18 : training on 3109 raw words (1818 effective words) took 1.1s, 1622 effective words/s
    2019-10-17 20:35:02,580 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:35:02,617 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:35:02,623 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:35:03,692 : INFO : EPOCH 19 - PROGRESS: at 100.00% examples, 1661 words/s, in_qsize 0, out_qsize 1
    2019-10-17 20:35:03,695 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:35:03,697 : INFO : EPOCH - 19 : training on 3109 raw words (1808 effective words) took 1.1s, 1652 effective words/s
    2019-10-17 20:35:03,705 : WARNING : this function is deprecated, use smart_open.open instead
    2019-10-17 20:35:03,735 : INFO : worker thread finished; awaiting finish of 2 more threads
    2019-10-17 20:35:03,738 : INFO : worker thread finished; awaiting finish of 1 more threads
    2019-10-17 20:35:04,777 : INFO : EPOCH 20 - PROGRESS: at 100.00% examples, 1735 words/s, in_qsize 0, out_qsize 1
    2019-10-17 20:35:04,779 : INFO : worker thread finished; awaiting finish of 0 more threads
    2019-10-17 20:35:04,782 : INFO : EPOCH - 20 : training on 3109 raw words (1824 effective words) took 1.1s, 1725 effective words/s
    2019-10-17 20:35:04,788 : INFO : training on a 62180 raw words (36290 effective words) took 21.3s, 1704 effective words/s
    2019-10-17 20:35:04,794 : WARNING : under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay
    

    Wall time: 21.4 s
    

Можно нормализовать вектора, тогда модель будет занимать меньше RAM. Однако после этого её нельзя дотренировывать. Здесь используется L2-нормализация: вектора нормализуются так, что если сложить квадраты всех элементов вектора, в сумме получится 1.


```python
model_liza.init_sims(replace=True)
model_path = "liza.bin"

print("Saving model...")
model_liza.wv.save_word2vec_format(model_path, binary=True)
```

    2019-10-17 20:35:04,821 : INFO : precomputing L2-norms of word weight vectors
    2019-10-17 20:35:04,870 : INFO : storing 478x300 projection weights into liza.bin
    2019-10-17 20:35:04,876 : WARNING : this function is deprecated, use smart_open.open instead
    

    Saving model...
    

Смотрим, сколько в модели слов:


```python
print(len(model_liza.wv.vocab))
```

    478
    


```python
print(sorted([w for w in model_liza.wv.vocab]))
```

    ['анюта', 'армия', 'ах', 'барин', 'бедный', 'белый', 'берег', 'березовый', 'беречь', 'бесчисленный', 'благодарить', 'бледный', 'блеснуть', 'блестящий', 'близ', 'бог', 'богатый', 'большой', 'бояться', 'брать', 'бросать', 'бросаться', 'бывать', 'быть', 'важный', 'ввечеру', 'вдова', 'велеть', 'великий', 'великолепный', 'верить', 'верно', 'весело', 'веселый', 'весна', 'вести', 'весь', 'весьма', 'ветвь', 'ветер', 'вечер', 'взглядывать', 'вздох', 'вздыхать', 'взор', 'взять', 'вид', 'видеть', 'видеться', 'видный', 'вместе', 'вода', 'возвращаться', 'воздух', 'война', 'воображать', 'воображение', 'воспоминание', 'восторг', 'восхищаться', 'время', 'все', 'вслед', 'вставать', 'встречаться', 'всякий', 'высокий', 'выть', 'выходить', 'глаз', 'глубокий', 'гнать', 'говорить', 'год', 'голос', 'гора', 'горе', 'горестный', 'горлица', 'город', 'горький', 'господь', 'гром', 'грусть', 'давать', 'давно', 'далее', 'дверь', 'движение', 'двор', 'девушка', 'дело', 'день', 'деньги', 'деревня', 'деревянный', 'десять', 'добро', 'добрый', 'довольно', 'доживать', 'долго', 'должный', 'дом', 'домой', 'дочь', 'древний', 'друг', 'другой', 'дуб', 'думать', 'душа', 'едва', 'ехать', 'жалобный', 'желание', 'желать', 'жениться', 'жених', 'женщина', 'жестокий', 'живой', 'жизнь', 'жить', 'забава', 'заблуждение', 'забывать', 'завтра', 'задумчивость', 'закраснеться', 'закричать', 'заря', 'здешний', 'здравствовать', 'зеленый', 'земля', 'златой', 'знать', 'ибо', 'играть', 'идти', 'имя', 'искать', 'исполняться', 'испугаться', 'история', 'исчезать', 'кабинет', 'казаться', 'какой', 'капля', 'карета', 'карман', 'картина', 'катиться', 'келья', 'клятва', 'колено', 'копейка', 'который', 'красота', 'крест', 'крестьянин', 'крестьянка', 'кровь', 'кроме', 'кто', 'купить', 'ландыш', 'ласка', 'ласковый', 'левый', 'лес', 'лететь', 'летний', 'лето', 'лиза', 'лизин', 'лизина', 'лицо', 'лишний', 'лодка', 'ложиться', 'луг', 'луч', 'любезный', 'любить', 'любовь', 'лютый', 'матушка', 'мать', 'место', 'месяц', 'мечта', 'милый', 'мимо', 'минута', 'многочисленный', 'могила', 'мой', 'молить', 'молиться', 'молния', 'молодой', 'молодость', 'молчать', 'монастырь', 'море', 'москва', 'москва-река', 'мочь', 'мрак', 'мрачный', 'муж', 'мы', 'мысль', 'наглядеться', 'надеяться', 'надлежать', 'надобно', 'называть', 'наступать', 'натура', 'находить', 'наш', 'небесный', 'небо', 'невинность', 'невинный', 'неделя', 'нежели', 'нежный', 'незнакомец', 'некоторый', 'непорочность', 'неприятель', 'несколько', 'никакой', 'никто', 'новый', 'ночь', 'обижать', 'облако', 'обманывать', 'обморок', 'образ', 'обращаться', 'обстоятельство', 'объятие', 'огонь', 'один', 'однако', 'окно', 'окрестности', 'он', 'она', 'они', 'оно', 'опираться', 'описывать', 'опустеть', 'освещать', 'оставаться', 'оставлять', 'останавливать', 'останавливаться', 'отвечать', 'отдавать', 'отец', 'отечество', 'отменно', 'отрада', 'очень', 'падать', 'память', 'пастух', 'первый', 'перемениться', 'переставать', 'песня', 'петь', 'печальный', 'писать', 'питать', 'плакать', 'побежать', 'побледнеть', 'погибать', 'подавать', 'подгорюниваться', 'подле', 'подозревать', 'подымать', 'поехать', 'пойти', 'показываться', 'поклониться', 'покойный', 'покрывать', 'покрываться', 'покупать', 'полагать', 'поле', 'помнить', 'поселянин', 'последний', 'постой', 'потуплять', 'поцеловать', 'поцелуй', 'правый', 'представляться', 'прежде', 'преклонять', 'прекрасный', 'прелестный', 'приводить', 'прижимать', 'принадлежать', 'принуждать', 'природа', 'приходить', 'приятно', 'приятный', 'провожать', 'продавать', 'проливать', 'простой', 'просыпаться', 'проходить', 'проч', 'прощать', 'прощаться', 'пруд', 'птичка', 'пылать', 'пять', 'работа', 'работать', 'радость', 'рассказывать', 'расставаться', 'рвать', 'ребенок', 'река', 'решаться', 'робкий', 'роза', 'розовый', 'роман', 'российский', 'роща', 'рубль', 'рука', 'сам', 'самый', 'свет', 'светиться', 'светлый', 'свидание', 'свирель', 'свободно', 'свое', 'свой', 'свойство', 'сделать', 'сделаться', 'сей', 'сердечный', 'сердце', 'сидеть', 'сие', 'сиять', 'сказать', 'сказывать', 'сквозь', 'скорбь', 'скоро', 'скрываться', 'слабый', 'слеза', 'слезать', 'слово', 'случаться', 'слушать', 'слышать', 'смерть', 'сметь', 'смотреть', 'собственный', 'соглашаться', 'солнце', 'спасать', 'спокойно', 'спокойствие', 'спрашивать', 'стадо', 'становиться', 'стараться', 'старуха', 'старушка', 'старый', 'статься', 'стена', 'сто', 'столь', 'стон', 'стонать', 'сторона', 'стоять', 'страшно', 'страшный', 'судьба', 'схватывать', 'счастие', 'счастливый', 'сын', 'таить', 'такой', 'твой', 'темный', 'тения', 'тихий', 'тихонько', 'томный', 'тот', 'трава', 'трепетать', 'трогать', 'ты', 'убивать', 'уверять', 'увидеть', 'увидеться', 'удерживать', 'удивляться', 'удовольствие', 'узнавать', 'улица', 'улыбка', 'уметь', 'умирать', 'унылый', 'упасть', 'услышать', 'утешение', 'утро', 'хижина', 'хлеб', 'ходить', 'холм', 'хороший', 'хотеть', 'хотеться', 'хотя', 'худо', 'худой', 'царь', 'цветок', 'целовать', 'час', 'часто', 'человек', 'чистый', 'читатель', 'чувствительный', 'чувство', 'чувствовать', 'чулок', 'шестой', 'шум', 'шуметь', 'щадить', 'щека', 'эраст', 'эрастов', 'это', 'я']
    

И чему же мы ее научили? Попробуем оценить модель вручную, порешав примеры. Несколько дано ниже, попробуйте придумать свои.


```python
model_liza.wv.most_similar(positive=["смерть", "любовь"], negative=["печальный"], topn=1)
```




    [('свой', 0.9896478056907654)]




```python
model_liza.wv.most_similar("любовь", topn=3)
```




    [('свой', 0.9973198175430298),
     ('лиза', 0.9972617626190186),
     ('сей', 0.9968301057815552)]




```python
model_liza.wv.similarity("лиза", "эраст")
```




    0.9977509




```python
model_liza.wv.similarity("лиза", "лиза")
```




    0.99999994




```python
model_liza.wv.doesnt_match("скорбь грусть слеза улыбка".split())
```




    'грусть'




```python
model_liza.wv.words_closer_than("лиза", "эраст")
```




    ['свой',
     'который',
     'мочь',
     'сказать',
     'сей',
     'сердце',
     'мой',
     'любить',
     'мать',
     'рука',
     'друг',
     'часто',
     'один',
     'душа',
     'смотреть',
     'лизин',
     'взять',
     'чистый',
     'берег']



#### Как использовать готовую модель

#### RusVectōrēs

На сайте RusVectōrēs (https://rusvectores.org/ru/) собраны предобученные на различных данных модели для русского языка, а также можно поискать наиболее близкие слова к заданному, посчитать семантическую близость нескольких слов и порешать примеры с помощью «калькулятором семантической близости».

Для других языков также можно найти предобученные модели — например, модели [fastText](https://fasttext.cc/docs/en/english-vectors.html) и [GloVe](https://nlp.stanford.edu/projects/glove/)

Ещё давайте посмотрим на векторные романы https://nevmenandr.github.io/novel2vec/

#### Работа с моделью

Модели word2vec бывают разных форматов:

+ .vec.gz — обычный файл
+ .bin.gz — бинарник

Загружаются они с помощью одного и того же гласса `KeyedVectors`, меняется только параметр `binary` у функции `load_word2vec_format`.

Если же эмбеддинги обучены не с помощью word2vec, то для загрузки нужно использовать функцию `load`. Т.е. для загрузки предобученных эмбеддингов `glove`, `fasttext`, `bpe` и любых других нужна именно она.

Скачаем с RusVectōrēs модель для русского языка, обученную на НКРЯ образца 2015 г.


```python
urllib.request.urlretrieve("http://rusvectores.org/static/models/rusvectores2/ruscorpora_mystem_cbow_300_2_2015.bin.gz", "ruscorpora_mystem_cbow_300_2_2015.bin.gz")
```




    ('ruscorpora_mystem_cbow_300_2_2015.bin.gz',
     <http.client.HTTPMessage at 0xb9a1da0>)




```python
m = 'ruscorpora_mystem_cbow_300_2_2015.bin.gz'

if m.endswith('.vec.gz'):
    model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=False)
elif m.endswith('.bin.gz'):
    model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=True)
else:
    model = gensim.models.KeyedVectors.load(m)
```


```python
words = ['хороший_A', 'плохой_A', 'ужасный_A', 'красный_A']
```

Частеречные тэги нужны, поскольку это специфика скачанной модели - она была натренирована на словах, аннотированных их частями речи (и лемматизированных). NB! В названиях моделей на `rusvectores` указано, какой тегсет они используют (mystem, upos и т.д.)

Попросим у модели 10 ближайших соседей для каждого слова и коэффициент косинусной близости для каждого:



```python
for word in words:
    # есть ли слово в модели? 
    if word in model:
        print(word)
        # смотрим на вектор слова (его размерность 300, смотрим на первые 10 чисел)
        print(model[word][:10])
        # выдаем 10 ближайших соседей слова:
        for i in model.most_similar(positive=[word], topn=10):
            # слово + коэффициент косинусной близости
            print(i[0], i[1])
        print('\n')
    else:
        # Увы!
        print('Увы, слова "%s" нет в модели!' % word)
```

    хороший_A
    [ 0.00722357 -0.00361956  0.1272455   0.06584469  0.00709477 -0.02014845
     -0.02056034  0.01321563  0.13692418 -0.09624264]
    плохой_A 0.7463520765304565
    неплохой_A 0.6708558797836304
    отличный_A 0.6633436679840088
    превосходный_A 0.6079519987106323
    замечательный_A 0.586450457572937
    недурной_A 0.5322482585906982
    отменный_A 0.5168066024780273
    прекрасный_A 0.4982394576072693
    посредственный_A 0.49099433422088623
    приличный_A 0.48622459173202515
    
    
    плохой_A
    [-0.05218472  0.0307817   0.1459371   0.0151835   0.06219714  0.01153753
     -0.01169093  0.01818374  0.0955373  -0.10191503]
    хороший_A 0.7463520765304565
    дурной_A 0.6186875700950623
    скверный_A 0.6014161109924316
    отличный_A 0.5226833820343018
    посредственный_A 0.5061031579971313
    неважный_A 0.5021153092384338
    неплохой_A 0.49169063568115234
    никудышный_A 0.48035895824432373
    ухудшать_V 0.43680471181869507
    плохо_ADV 0.4314875304698944
    
    
    ужасный_A
    [-0.05553271 -0.03172469  0.01998607  0.00171507 -0.00935555 -0.0296017
      0.05394973  0.01597532 -0.03785459 -0.02099892]
    страшный_A 0.8007249236106873
    жуткий_A 0.6982528567314148
    отвратительный_A 0.6798903942108154
    ужасающий_A 0.6174499988555908
    чудовищный_A 0.6100855469703674
    постыдный_A 0.6009703874588013
    невероятный_A 0.5827823281288147
    ужасать_V 0.5815353393554688
    кошмарный_A 0.5675789713859558
    позорный_A 0.5351496338844299
    
    
    красный_A
    [ 0.01627072 -0.01136785 -0.00790482  0.02294072  0.05129128  0.10162549
      0.07488654 -0.06475785 -0.0203686   0.09159683]
    алый_A 0.642128586769104
    малиновый_A 0.6113020777702332
    красная_S 0.5526680946350098
    желтый_A 0.5431625247001648
    оранжевый_A 0.5371882319450378
    трехцветный_A 0.531793475151062
    пунцовый_A 0.5125025510787964
    синий_A 0.5102002024650574
    фиолетовый_A 0.5072877407073975
    лиловый_A 0.5004072785377502
    
    
    

Находим косинусную близость пары слов:


```python
print(model.similarity('плохой_A', 'хороший_A'))
```

    0.7463521
    

Пропорция

+ positive — вектора, которые мы складываем
+ negative — вектора, которые вычитаем


```python
print(model.most_similar(positive=['плохой_A', 'ужасный_A'], negative=['хороший_A'])[0][0])
```

    страшный_A
    

Найди лишнее!


```python
print(model.doesnt_match('плохой_A хороший_A ужасный_A страшный_A'.split()))
```

    хороший_A
    


```python
for word, score in model.most_similar(positive=['ужасно_ADV'], negative=['плохой_A']):
    print(f'{score:.4}\t{word}')
```

    0.5575	безумно_ADV
    0.4791	безмерно_ADV
    0.4536	жутко_ADV
    0.4472	невероятно_ADV
    0.4394	очень_ADV
    0.4364	чертовски_ADV
    0.4231	страшно_ADV
    0.4124	необычайно_ADV
    0.4119	нестерпимо_ADV
    0.4005	необыкновенно_ADV
    

#### Оценка

Это, конечно, хорошо, но как понять, какая модель лучше? Или вот, например, я сделал свою модель, а как понять, насколько она хорошая?

Для этого существуют специальные датасеты для оценки качества дистрибутивных моделей. Основных два: один измеряет точность решения задач на аналогии (про Россию и пельмени), а второй используется для оценки коэффициента семантической близости.

#### Word Similarity

Этот метод заключается в том, чтобы оценить, насколько представления о семантической близости слов в модели соотносятся с \"представлениями\" людей.

| слово 1    | слово 2    | близость |
|------------|------------|----------|
| кошка      | собака     | 0.7      | 
| чашка      | кружка     | 0.9      | 

Для каждой пары слов из заранее заданного датасета мы можем посчитать косинусное расстояние, и получить список таких значений близости. При этом у нас уже есть список значений близостей, сделанный людьми. Мы можем сравнить эти два списка и понять, насколько они похожи (например, посчитав корреляцию). Эта мера схожести должна говорить о том, насколько модель хорошо моделирует расстояния о слова.

#### Аналогии

Другая популярная задача для "внутренней" оценки называется задачей поиска аналогий. Как мы уже разбирали выше, с помощью простых арифметических операций мы можем модифицировать значение слова. Если заранее собрать набор слов-модификаторов, а также слов, которые мы хотим получить в результаты модификации, то на основе подсчёта количества "попаданий" в желаемое слово мы можем оценить, насколько хорошо работает модель.

В качестве слов-модификатор мы можем использовать семантические аналогии. Скажем, если у нас есть некоторое отношение "страна-столица", то для оценки модели мы можем использовать пары наподобие "Россия-Москва", "Норвегия-Осло", и т.д. Датасет будет выглядеть следующм образом:

| слово 1    | слово 2    | отношение     | 
|------------|------------|---------------|
| Россия     | Москва     | страна-столица| 
| Норвегия   | Осло       | страна-столица|

Рассматривая случайные две пары из этого набора, мы хотим, имея триплет (Россия, Москва, Норвегия) хотим получить слово "Осло", т.е. найти такое слово, которое будет находиться в том же отношении со словом "Норвегия", как "Россия" находится с Москвой.

Датасеты для русского языка можно скачать на странице с моделями на RusVectores. Посчитаем качество нашей модели НКРЯ на датасете про аналогии:


```python
res = model.accuracy('ru_analogy_tagged.txt')
```

    2019-10-17 14:06:20,510 : INFO : capital-common-countries: 19.0% (58/306)
    2019-10-17 14:06:24,401 : INFO : capital-world: 10.1% (52/515)
    2019-10-17 14:06:25,328 : INFO : currency: 4.6% (6/130)
    2019-10-17 14:06:27,458 : INFO : family: 71.2% (218/306)
    2019-10-17 14:06:33,288 : INFO : gram1-Aective-to-adverb: 18.7% (152/812)
    2019-10-17 14:06:35,937 : INFO : gram2-opposite: 32.1% (122/380)
    2019-10-17 14:06:42,301 : INFO : gram6-nationality-Aective: 32.3% (293/907)
    2019-10-17 14:06:42,303 : INFO : total: 26.8% (901/3356)
    


```python
for row in res[4]['incorrect'][:10]:
    print('\t'.join(row))
```

    МАЛЬЧИК_S	ДЕВОЧКА_S	ДЕД_S	БАБКА_S
    МАЛЬЧИК_S	ДЕВОЧКА_S	КОРОЛЬ_S	КОРОЛЕВА_S
    МАЛЬЧИК_S	ДЕВОЧКА_S	ПРИНЦ_S	ПРИНЦЕССА_S
    МАЛЬЧИК_S	ДЕВОЧКА_S	ОТЧИМ_S	МАЧЕХА_S
    МАЛЬЧИК_S	ДЕВОЧКА_S	ПАСЫНОК_S	ПАДЧЕРИЦА_S
    БРАТ_S	СЕСТРА_S	ДЕД_S	БАБКА_S
    БРАТ_S	СЕСТРА_S	ОТЧИМ_S	МАЧЕХА_S
    БРАТ_S	СЕСТРА_S	ПАСЫНОК_S	ПАДЧЕРИЦА_S
    ПАПА_S	МАМА_S	ДЕД_S	БАБКА_S
    ПАПА_S	МАМА_S	ОТЧИМ_S	МАЧЕХА_S
    

**Визуализация**

Можно использовать разные методы того, как преобразовать векторы так, чтобы можно было их поместить на двумерное пространство, например, с помощью PCA. В зависимости от того, относительно какого набора слов вы пытаетесь найти оптимально отображение на двумерное пространство, у вас могут получаться разные результаты


```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
words = ['кошка_S', 'собака_S', 'корова_S', 'коза_S', 'бык_S']
X = model[words]
```

На списке конкретных слов


```python
pca = PCA(n_components=2)
coords = pca.fit_transform(X)
```


```python
plt.scatter(coords[:, 0], coords[:, 1], color='red')
plt.title('Words')

for i, word in enumerate(words):
    plt.annotate(word, xy=(coords[i, 0], coords[i, 1]))
plt.show()
```


![png](output_47_0.png)


На все словах в модели


```python
pca = PCA(n_components=2)
pca.fit(model[list(model.vocab)])
coords = pca.transform(model[words])
```


```python
plt.scatter(coords[:, 0], coords[:, 1], color='red')
plt.title('Words')

for i, word in enumerate(words):
    plt.annotate(word, xy=(coords[i, 0], coords[i, 1]))
plt.show()
```


![png](output_50_0.png)

