# __TIL__ 

## 2020.11.11

### 파이썬 요약 강의
+ print( sep=' ' ) 
+ False : '', 0 , None 
+ True : 값이 있는 모든 것 
+ and = * , or = +
+ 비트 연산자 : & , | , ~ 
+ 이진법으로 변환 후 & => * , | => + 연산


* def function(x) :  x => parameter
* function(4) : 4 => argument

```python
a=10 
def function() : 
    a+=10 
    return a      # 에러가 뜬다 , 왜 ? a는 함수 내의 지역변수로, 선언되어 있지 않기 때문 !
def function2() :
    global a
    a+=10
    return a      # 함수 밖의 전역변수 a를 가져와서, 20을 올바르게 return하게 된다 !
```

+ 리스트 , 튜플은 순서가 있는 자료형 !
+ 셋 , 딕셔너리는 순서가 없는 자료형 !


```python
a = 1
while True :
    print('while 루프')
    a+=1
    if a>10 :
        break 
else :
    print('정상종료')  # while문이 break에 의해 종료되어 출력되지 않음 ! 

a = 1
while a<10 :
    print('while 루프')
    a+=1
else :
    print('정상종료') # while문의 조건에 의해 정상종료되었으므로, else 문 출력 !

#for 문에도 break 와 else ( 정상종료 ) 사용 가능
```
```python
s= {1,2,3}
d= {'one':1, 'two':2}     # 순서가 없는 자료형

for i in s :
    print(i)    # 순서는 없으나, 순회는 가능 !
for i in d :
    print(d)    # key만 순회한다 !

```


```python
# 언패킹 
l = [(1,10) , (2,20) , (3,30)]
for i in l :
    print(i)
for i , j in l  :
    print(i,j)
```
* continue : 다음 순회로 넘어가기
* pass : 아무 기능도 하지 않는다 !
* enumerate(x, n) : n - 시작숫자

### Numpy

```python

import numpy as np

data = [[1,2,3],[4,5,6],[7,8,9]]
a = np.array(data) 
a   # ndarray ~ 행렬의 형태 !
a.dtype
a.astype('float32') #형변환
a.dtype

np.arange(1,10).reshape(3,3)   #arange : range와 같음. reshape(A,B) : A x B 행렬로 생성 / 앞의 데이터와 사이즈 같아야함!

np.nan # 평균 등의 계산에서 자동 제외 ! / float형에서 사용 가능

b = np.linspace(1,10,20)  # (초기값, 종료값, 생성개수)

```

### Numpy 연산

```python
data = np.arange(1,10).reshape(3,3)
data + data # lis의 + 연신처럼 연결이 아닌 , 각 원소끼리의 사칙연산 
np.dot(data,data) # 행렬의 곱연산 
data@data #상동
```
### Numpy 차원
차원수 == 대괄호의 갯수

```python
#0차원 : 스칼라값

a = np.array(1)
a
a.shape
a.ndim #차원 반환

#1차원 : 벡터

a = np.array([1])
a
a.shape
a.ndim #차원 반환

a = np.array([1,2,3,4,5])
a
a.shape 
a.ndim #차원 반환

#2차원 : 매트릭스 (행렬)
a = np.array([[1,2,3],[4,5,6]])
a
a.shape # 행과 열의 수
a.ndim #차원 반환

#3차원 : 3차원 이상의 다차원 행렬 = Tensor
a = np.array([[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]])
```

```python
a = np.ones(12)
b = np.zeros(12)
c = np.eye(3) # n by n 의 단위행렬

d = a.reshape(2,3)
e = np.ones([2,3]) #이렇게 바로 만들수도 있음

f = np.empty([2,3]) #0에 가까운, 값이 있는 값!
g = np.full((2,3), 1000) #

h = np.linspace(2, 10, 6) #2부터 10까지 6개를 동등한 간격으로 !
```
### Numpy 집계함수
```python
a= np.arange(10).reshape(2,5)
a[0][0] 
a[0,0]
np.mean(a)
np.median(a)
np.std(a)
np.var(a)

np.sum(a) # 모든 원소의 합

sum(a) # column 별 합
np.sum(a, axis=0) # column 별 합
np.sum(a, axis=1) # row 별 합
```


### Pandas

+ 
```python 
import pandas as pd
import numpy as np

data = np.arange(0, 50, 10)

a = pd.Series(data, index=['a','b','c','d','e']) # a, b, c, d, e

a['b'] 
a.loc['b']  # 인덱스 이름 : 명시적 인덱스
a.iloc[1] #순서 : 암시적 인덱스
```
+ Dataframe 
```python 
rawdata = np.random.randint(50,100,size=(4,3))
df = pd.DataFrame(rawdata,index=['1반','2반','1반','2반'],columns=['국','영','수'])
df['국'] # columns 값 먼저 ! df[열][행]
#df[0] -> Error
df.dropna(axis = 0 , inplace= True) # inplace : 원본 수정!
```

### Dataframe Indexing

```python
df.T # row와 column 을 바꿔준다 !

df.index = [['1학년','1학년','2학년','2학년'],['1반','2반','1반','2반']] #2차월 행렬을 통해 MultiIndexing
```

```python  

a = pd.DataFrame(np.arange(1,10.reshape(3,3)))
b = pd.Series(np.arange(10,40,10))
pd.concat([a,b], axis =1 ,ignore_index= True)  # a에 b를 옆으로 (열로 ) 연결, ignore_index : b의 기존 인덱스 무시
a.append(b) # a에 b를 밑으로 연결 
```

## 2020.11.15

### Matplotlib 

```python 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

x = [100,200,300]
y = [1,2,3]

value = pd.Series(y,x) #y 값 먼저 !

plt.plot(x,y)
plt.plot(value)
plt.plot(value, '--') # ':' , '--r', '--o' 점선, 색상, 동그라미
plt.plot(value, color='red') #ff0000 과 같이 16진법 RGB로 표현 가능
plt.plot(value, linewidth= 10)
plt.title('hello world', fontsize=20)
plt.xlabel('hello', fontsize=20)
plt.ylabel('world', fontsize=20)
plt.savefig('sample.png')

x=np.linspace(0,10,100)
y=np.sin(x)
y_=np.cos(x)

plt.plot(x,y, label = 'sin' )
plt.plot(x,y_,'-o', label = 'cos' )  #legend 가 있어야 label 출력됨

plt.legend(loc=4) #loc로 legend의 위치 조정
```

### Matplotlib_Scatter

```python 
x = np.linspace(0,10,20)
y = x**2
plt.scatter(x,y, c ='r' , alpha = 0.5) #c==color , alpha = ?(색상관련)
plt.show()
```

### Matplotlib_Histogram

```python 
x = [np.random.randint(1,7) for i in range(1000)]
plt.hist(x,bins=6) # bins = 칸수 조정
plt.show
```

### Matplotlib_Piechart

```python 
labels = ['one','two','three' ]
size = [100,20,30]
plt.pie(size,labels=labels, autopct = '%1.2f%%')
```
### Matplotlib_Barchart

```python 

plt.bar(['one','two','three'],[10,20,30])
plt.barh(['one','two','three'],[10,20,30])
```

### Plotly

[Plotly 튜토리얼 공식 문서](https://plotly.com/python/)
```python
import plotly.express as px
import plotly.graph_object as go
x = [1,2,3,4,5]
y = x**2
fig = px.line(x=x,y=y)
fig.show()

korea_life = px.data.gapminder().query("country == 'Korea, Rep.'")
fig = px.line(korea_life, x = "year", y="lifeExp", title='Life expectancy in Korea')
fig.show()


korea_GDP = px.data.gapminder().query("country == 'Korea, Rep.'")
fig = px.bar(korea_GDP, x = "year", y="gdpPercap", title='한국인 GDP')
fig.show()


korea_GDP = px.data.gapminder().query("country == 'Korea, Rep.'")
fig = px.scatter(korea_GDP, x = "year", y="gdpPercap", title='한국인 GDP')
fig.show()


fig = px.pie(values = [20, 30, 50])
fig.show()


```

### 이미지 분석
```python 
import numpy as np
from skimage import io
# from PIL import Image
# from cv2
import matplotlib.pyplot as plt
jeju = io.imread('jeju.jpg')
type(jeju)
jeju.shape
jeju
np.min(jeju), np.max(jeju)
plt.imshow(jeju)
plt.imshow(jeju[::-1]) #상하좌우반전      [strat, stop, step]
plt.imshow(jeju[:, ::-1]) #좌우반전
plt.imshow(jeju[::-1,:]) #상하반전

plt.imshow(jeju[500:800,:])
plt.imshow(jeju[500:800,500:730]) # 말 이미지만 슬라이싱

plt.imshow(jeju[::3,::3]) #3칸씩 건너 뛰기
minijeju = jeju[::3,::3]

plt.hist(minijeju.ravel(),256,[0,256]) # minijeju의 색의 분포
plt.show

minijeju_ = np.where(minijeju < 50, minijeju , 0) #minijeju 에서 , 50 이하의 값은 minijeju, 그렇지 않은 값 = 0
plt.imshow(minijeju_)

plt.imshow(jeju[:,:,1])

from skimage import color

plt.imshow(color.rgb2gray(jeju),cmap=plt.cm.gray)
```


## 2020.11.17

### 객체, 복사
```python

NEW = SLL()
print('NEW = ', id(NEW))
print('NEW.head = ', id(NEW.head))
x =Node(0)
print('x (Node) = ', id(x))
print('None =', id(None))

NEW.head = x 
print('NEW.head = ', id(NEW.head))
print('x (Node) = ', id(x))

x = Node(1)
print('NEW.head = ', NEW.head, id(NEW.head))
print('x (Node) = ', id(x))

## 인스턴스 변수가 가르키던 변수 x의 값이 변해도, 기존 인스턴스 변수에 배정된 주소 (바뀌기 전 x) 유지 ! x가 가르키는 주소만 바뀜 !
## 위의 예시에서 인스턴스 변수는 고유한 주소가 아니라 , 초기에 빈 공간 None을 가지며, 
## x가 배정된 뒤에야 x가 가르키는 주소를 가르킴

a= 1
b=a # b에 a가 대입되는 것이 아니라, a가 가르키는 저장공간이 가진 값을 불러와서 b에 대입하는 것
print(id(a))
print(id(b))
a=2


print(a)
print(b)

print(id(a))
print(id(b))

## a의 주소가 1이라는 객체의 주소를 버리고, 2라는 새로운 객체의 주소를 가르키게 되는 것!
## 따라서, b = a 였다고 해서 b가 가르키는 객체가 변하거나 하지 않음.
## b는 그대로 1을 가르키고, a의 주소만 이동

## 변수가 저장 공간이 아님 ! 저장공간을 가르키는 이름표일 뿐 ! 
## 기존 저장공간에서 이름표를 떼어서 다른 곳에 붙이는 것!

a= [1,2,3]
b=a
print(id(a))
print(id(b))
a=[4,5,6]


print(a)
print(b)

print(id(a))
print(id(b))


## 이 경우에도 위와 같으나 ,

a= [1,2,3]
b=a
print(id(a))
print(id(b))
a[0] = 10


print(a)
print(b)

print(id(a))
print(id(b))

## 이처럼 a가 가르키는 주소가 바뀌는 것이 아니라, 여전히 a와 b는 같은 주소를 가르키고 있고,
## 해당 주소의 객체 자체가 변하는 경우,
## a에서 객체에 변화를 준 것이 b 에서도 나타나게 됨!

def func(s) :                   # s = st    s가 st 변수에 저장된 값을 가르키게 됨.
                                # 즉, [1,2,3] 에 st란 이름표와 s란 이름표가 두개 붙은 상태!
    s[0] = 0                    # s란 이름표를 통해 [1,2,3]에 접근해 0번 인덱스를 변경


st = [1,2,3]                    
func(st)
>>> st == [0,2,3]
```

## 2020.11.19

### Iterator, Generator, yield

+ iterator : 한 번에 하나씩 그 객체의 elements에 순서대로 액세스 할 수 있는 객체

    + 'iterable' 한 객체가 iter() 함수를 통하여 'iterator 객체'로 생성됨
    + iter(iterable) -> iterator
        - cf) iterator도 iterable 자리에 올 수 있다. 하지만 자기 자신을 그대로 반환 !


    - iter() , next() 지원
    + iter(f) 가 정의된 f.\_\_iter__() 메소드를 호출 : next 메소드를 가지는 iterator 객체를 반환
    + next(f) 가 정의된 f.\_\_next__() 메소드를 호출 : 반복자를 입력받아 다음 요소 반환
    
+ generator 는 iterator의 특수한 형태 
    + generator : yield 를 통하여 next가 호출될 때 마다 다음 값 반환 
    + yield 명령어로 iterator를 만드는 함수


+ return : 값을 함수 외부로 전달하고, 함수를 종료
+ yield : 값만 함수 외부로 전달하고, 함수를 종료하지 않음 (함수의 일시정지)
    + 다음 next()가 호출될 때까지 대기 ?  
    + 값을 미리 만들어두지 않고, next가 호출될 때만 불러오기 때문에 메모리 효율 UP      
       
_____



+ for문은 순회하기위해 제일먼저 \_\_iter__() 메서드를 호출한다. 
+ 그리고 for문이 반복할때마다 \_\_next__() 메서드를 호출하여 다음 값으로 넘어간다. 
+ 생성자에서 i 변수를 만들고, \_\_iter__메서드에서는 self를 return 한다. 그리고 \_\_next__ 메서드에서 i를 +1씩 증가 시킨다.   
  
    

_____



+ 제네레이터도 이터레이터랑 기능은 동일하다. 하지만 이터레이터보다 좀더 간편하게 기능을 구성할 수 있다. 
+ 제네레이터는 에터레이터처럼 \_\_iter__ 나 \_\_next__ 를 구현할 필요가 없다. 다만 함수안에 yield문을 사용하여 값을 반환하면된다.  
  

_____

```python
class It :
    def __init__(self) :
        self.i = 0
    def __iter__(self) :    # iterable한 객체 정의 
        return self         # self를 return 함으로써 이터레이터로 구현.
    def __next__(self) :    # iterator는 next( f )를 통해 f.__next__() 호출하여 다음 요소 불러옴.
        if self.i == 5 :
            raise StopIteration
        else :
            self.i += 1
            return self.i

k = It()


for i in k :
    print(i)
## for문에서 Iter() 메소드를 자동으로 불러와, iterator 생성, next() 호출

a = iter(k)     # iter객체를 생성하지 않으면 next 호출이 안되네
print(next(a))
print(next(a))
print(next(a))
print(next(a))
print(next(a))





class Gn :
    def __init__(self) :
        self.i = 0
    def gene(self) :
        while self.i < 5 :
            self.i +=1
            yield self.i
g=Gn()

for i in g.gene() : # gene() 메소드로 generator 생성, for 문에서 순회 가능한 객체! next() 필요 없이 generator 자체적으로 순회중
    print(i)

for i in g :    # Gn 클래스 자체가 Iterable 하지는 않음! gene 메소드를 통하여 generator를 생성하는 것이기 때문!
    print(i) 




class Gn2 :
    def __init__(self) :
        self.i = 0
    def __iter__(self) :    # generator를 생성하는 메소드를, __iter__로 정의해줌 !
        while self.i < 5 :
            self.i +=1
            yield self.i
g2=Gn2()
for i in g2 :   #   for문은 해당 클래스의 __iter__ 메소드를 불러온 후 순회하는 구조 ! generator를 __iter__ 메소드 안에 구현해줬기 때문에, 
    print(i)    #   generator를 별도롤 불러오지 않아도 객체에서 자동으로 !
```



>>> \>\>\> 를 쓰면 이렇게 되네 !




+ list와 set 같은 집합에 대한 iterator는 이미 모든 값들을 저장해 둔 상태이지만, generator는 모든 값들을 갖지 않은 상태(미정)에서 yield에 의해 하나씩만 데이터를 만들어 가져온다는 차이점이 있다.




+ iter는 반복을 끝낼 값을 지정하면 특정 값이 나올 때 반복을 끝냅니다. 이 경우에는 반복 가능한 객체 대신 호출 가능한 객체(callable)를 넣어줍니다. 참고로 반복을 끝낼 값은 sentinel이라고 부르는데 감시병이라는 뜻입니다. 즉, 반복을 감시하다가 특정 값이 나오면 반복을 끝낸다고 해서 sentinel입니다.

+ iter(호출가능한객체, 반복을끝낼값)
예를 들어 random.randint(0, 5)와 같이 0부터 5까지 무작위로 숫자를 생성할 때 2가 나오면 반복을 끝내도록 만들 수 있습니다. 이때 호출 가능한 객체를 넣어야 하므로 매개변수가 없는 함수 또는 람다 표현식으로 만들어줍니다.
```python
>>> import random
>>> it = iter(lambda : random.randint(0, 5), 2)
>>> next(it)
0
>>> next(it)
3
>>> next(it)
1
>>> next(it)
Traceback (most recent call last):
  File "<pyshell#37>", line 1, in <module>
    next(it)
StopIteration
```



## 2020.11.23

### Error 처리
+ try : 정상적으로 실행 되었을 때 !
+ except : try문에서 Error 가 발생했을 때 실행되는 블럭
+ else : 어떤 error도 발생하지 않은 경우 else 블럭 실행 ( Except로 예외처리 해둔 에러가 발생해도 Else는 실행되지 않는다.)
+ finally : try가 정상적으로 실행된 후든 , except문이 실행된 후든, except처리가 안된 에러가 발생한 경우든 finally 문은 실행됨!

### 상속
+ super() : 상속관계에서 부모클래스를 호출하는 키워드
    - ex/ super().메소드명()  
    - 오버라이딩을 따로 하지 않은 경우, 
    - 자식클래스에서 부모클래스의 생성자 호출 이후에
    - self.메소드명()로도 부모클래스 호출 가능
    


+ 오버라이딩 : 자식 클래스에서 부모 클래스의 함수 덮어쓰기 

+ 오버로딩 (JAVA) : 같은 이름의 메소드, 자료형이나 파라미터 개수가 다르면 이름 겸용 가능 !

### 모듈 
+ 필요할 떄 가져다 쓸 수 있는, 함수만을 담고 있는 부품의 개념
+ 담긴 내용이, 실행의 흐름이 아니라 다른 프로그램에서 가져다 쓸 수 있는 내용
+ 명확한 정의는 아님, 실행 함수가 있는 소스 파일들도 , 즉 모든 소스 파일을 모듈로 얘기하기도 함



+ import 모듈이름     / as 로 모듈명 바꾸기 ex) 모듈 이름 길 경우
    
    + 모듈이름.함수()
+ from 모듈이름 import 함수이름   / as 로 함수명 바꾸기 . ex) 함수명 충돌
    
    + 그냥 함수 이름 사용 !

+ 소스파일과 모듈을 같은 디렉토리에
    + 빌트인 모듈 : 디렉토리, 소스파일 상관 없이 언제나 import 가능한 모듈 / ex) math
    + 빌트인 함수 : 모듈 import 없이 호출 가능한 함수








## 2020.11.25

### 접근제한

+ 기본적으로 클래스 내부 변수는 public 성질, 외부에서 접근 및 변경 가능
+ __멤버변수 / __method() 형태로 선언 -> private 지정 
    - 외부에서 _클래스명__변수/메소드명() 으로 접근 가능하긴 함 !



## 2020.12.01

### 지역변수 / 전역변수

+ 지역변수 : 함수 안에서 선언된 변수. 함수 안에서 생성되고, 함수 안에서만 사용된 후 소멸됨. 
+ 전역변수 : 함수 밖에서 선언된 변수. 프로그램 전체에서 이용 가능함. 함수 안에서도 사용 가능!
    - 읽기 / 수정 ( append 등 ) : 함수 내에서 가능 
    - '=' 대입 : 함수 내에서 불가능 / global 사용하면 가능
+ global : 함수 내에서 선언해도 전역변수로 사용 가능
+ 매개변수 : 매개변수도 지역변수 !

```python 
def plus(num) :
    b = num + a # a를 선언하기 전에 사용해도 되나 ? oK
                # 단, global을 사용하지 않으면, 전역변수에 '=' 을 사용한 대입은 불가능 !
                # 단, 전역변수 참조 및 수정 ( append 등 ) 은 함수 내에서도 가능
    return b
a = 3

print(plus(5)) # a를 선언한 뒤 함수를 호출하니까 괜찮음 !
```
### *args / **kargs 

+ 가변인자 (*args) : 입력 인자가 변하는 함수 ( 여러 개 입력 가능 )

```python
def func(x, *nums) : # *args 자리에 들어오는 값들을 튜플의 형태로 받음 / 일반 매개변수 뒤에 위치해야함 !
    print('x =', x)
    print(nums)
```
+ 키워드 매개변수 (**kargs) :  입력 인자를 딕셔너리 형태로 받음 !

```python

def func2(x, **nums) : # **kargs 자리에 들어오는 값들을 딕셔너리의 형태로 받음 / 일반 매개변수 뒤에 위치해야함 !
                       # 정확히는, k1 = v1 , k2 = v2 형태로 받음
                       # 딕셔너리를 집어 넣으면 에러.
                       # 딕셔너리를 k = v 형태로 풀어준 뒤 입력해야함...!

    print('x =', x) # 일반 매개변수
    print(nums)

    


func2('x', k=1,m=2,n=3) # Key = Value 의 형태로 전달

dic = {'p':4}      
>>> func2('x', dic)     # ERROR 

>>> func2('x', **dic)   # **를 붙여서 언패킹한 형태로 전달해야함.
                        # 함수 정의시의 **가 애초에 입력된 값을 딕셔너리로 묶는 것이기 때문에,
                        # 딕셔너리를 딕셔너라로 묶는 것이 에러 !

>>> func2('x', k=1,m=2,n=3, **dic)  # 혼용도 가능 !
```


### 객체의 비교와 복사 
+ v1 == v2  : 내용 비교
+ v1 is v2  : 동등 비교 (주소값)



```python 

r1 = ['john',('man','USA'),[175,60]]
r2 = list(r1)       # r1을 복사하여 r2에 대입 / r1과 r2는 다른 객체! 

r1 is r2  # False        

r1[0] is r2[0]  # True
r1[1] is r2[1]  # True
r1[2] is r2[2]  # True

# 큰 틀은 다른 객체로 복사가 되지만, 복사한 객체 (리스트) 내의 값들은 같은 주소를 참조하게 복사됨 ! 
# 즉, 서로 다른 객체지만 같은 값의 객체를 공유하고 있는 상태
# r2에서 [2] 리스트 값을 변경하면 r1[2]도 변경됨 

# 불변 값의 경우 공유해도 상관 없겠지만, 가변값의 경우 공유하면 문제 발생
```
>>>얕은 복사

```python 
import copy

r1 = ['john',('man','USA'),[175,60]]
r2 = copy.deepcopy(r1)       # copy 모듈의 deepcopy / 효율과 성능을 위해 가변값만 깊은 복사 !

r1 is r2  # False        

r1[0] is r2[0]  # True / 얕은복사
r1[1] is r2[1]  # True / 얕은복사
r1[2] is r2[2]  # False / 깊은복사

r2[2][1] += 5

r1[2]
r2[2]

# [0],[1] 의 경우는 불변값이므로, 같은 대상을 참조하여도 별 문제가 없음 -> 얕은복사
# [2] 의 경우는 가변값이므로, 새로운 객체를 복사하여 따로 참조 -> 깊은복사


```
>>>깊은 복사

### 함수도 객체이다 ?
+ 함수도 객체이기 때문에, 매개변수로 전달될 수도 있고, return으로 반환될 수도 있다 !

```python
def caller(fct) : # 함수 객체를 fct로 참조하여, 호출하는 함수
    fct()

def say() :
    print('Hello world!')

caller(say)




def show(n) :
    print(n)

ref = show # show 함수 객체를 ref 도 참조하게 한다. 
           # 정확히는, show 가 참조하는 함수 객체를 ref 도 참조하게 하는 것.
ref('hello') 

```
>>> 함수도 객체임을 알 수 있다  

>>> 그렇다면 함수에 이름이 필요한 이유 ? 함수 객체를 참조하고 호출하기 위한 변수명일 뿐 !

```python 

def func(n):
    def exp(x) :
        return x ** n 
    return exp 


f2 = func(2) # x**2 를 반환하는 exp(x) 객체를 반환, f2가 이를 참조 
f2(4) # 16
f3 = func(3) # x**3 을 반환하는 exp(x) 객체를 반환, f3가 이를 참조
f3(4) # 64
```

+ lambda  :  lambda 매개변수 : 함수몸체 
    - 위의 예에서, 함수명은 함수 객체를 참조하기 위한 변수명일 뿐을 확인함
    - 변수명이 필요 없는 경우는 ?
```python

def show(n) :
    print(n)

ref = show # show 함수 객체를 ref 도 참조하게 한다. 
           # 정확히는, show 가 참조하는 함수 객체를 ref 도 참조하게 하는 것.
ref('hello') 

# 위의 이 예시에서, 같은 객체의 레퍼런스 카운트가 2 이므로, 굳이 show 라는 함수명을 사용하지 않기로 한다.

lambda x : print(x)         # 이와 같이 함수명 없이 매개변수/함수몸체로만 객체를 구성할 수 있다.
type(lambda x : print(x)) 



ref = lambda x : print(x) # 하지만 위와 같이 선언하면, 레퍼런스 카운트가 0이므로 , ref 변수명이 참조하게 한다 !
ref('hello') 


f = lambda x : x+2 # return 명시 없이도 자동으로 return 하게 됨.
f(4)
```
>>> 함수는 기본적으로 함수명 / 매개변수 / 함수몸체로 구성되지만, 함수명을 생략하는 것 !

### map & filter 

+ map(함수, iterable) 
    - iterable한 객체의 값에 하나씩 인자로 입력받은 함수를 적용
    - 적용한 결과를 저장한 iterator 객체를 반환 !


```python
a = map(lambda x : x*2 , [1,2,3])     # a 는 iterator 객체
print(next(a))                        # 2
print(next(a))                        # 4
print(next(a))                        # 6
b = list(a)                           # [2,4,6]
```

```python
def sum(x,y) :
    return x+y

a = map(sum, [1,2,3], [10,20,30])      # 처음 전달받은 iterable[0] , 다음 전달받은 iterable[0] 을 sum, [1]과 [1] sum ....
a = list(a)
print(a)
```
>>> 전달된 함수의 인자가 여러개인 경우, 그 개수만큼 iterable 입력


+ filter : filter(함수, iterable)
    - 값을 걸러내는 기능 
    - 인자로 전달받은 함수의 에 iterable 객체의 값을 하나씩 전달
    - 그 함수의 결과 return 값이 True 인 값들만 저장
    - iterator 반환
```python

st = [1,2,3,4,5]
filter(lambda x : x%2, st)                      # x%2 == 1 (True) 인 값만 담은 iterator
filter(lambda n : not(n % 3), range(1,11) )     # n%3 == 0 (False) 인 값만 담은 iterator / not False == True
filter(lambda n : not(n % 3), map(lambda x : x ** 2, range(1,11)))
# 1~10 의 제곱수 중, 3의 배수만 담은 iterator
```

## 2020.12.02

### 제너레이터 표현식

+ 리스트 컴프리헨션과 같은 용법 
    - (n ** 2 for n in range(1,11))
    - [] 대신 () 을 사용하면 제너레이터 객체가 생성됨 !

```python
class Gn :
    
    def __iter__(self) :    # generator를 생성하는 메소드를, __iter__로 정의해줌 !
        for i in range(5) :
            yield i
g=Gn()

for i in g :   #   for문은 해당 클래스의 __iter__ 메소드를 불러온 후 순회하는 구조 ! generator를 __iter__ 메소드 안에 구현해줬기 때문에, 
    print(i)    #   generator를 별도롤 불러오지 않아도 객체에서 자동으로 !


class Gn2 :
  
    def __iter__(self) :    # generator를 생성하는 메소드를, __iter__로 정의해줌 !
        return (i for i in range(5))

g2 = Gn(2)

for i in g2 :
    print(i)
```

+ 제너레이터 표현식을 인자로 전달 
```python
def show(x) :
    for i in x :
        print(i)

g = (i*2 for i in range(5))
show(g)                         # 이렇게 변수명으로 전달할 수 있지만
show((i*2 for i in range(5)))   # 직접 인자로 넣을 수도 있음.
show(i*2 for i in range(5)      # 이 경우, 표현식의 () 를 생략해도 되도록 약속!
```

### 튜풀 패킹 & 언패킹 
+ 튜플 패킹 
    - '__*__' 사용하여 패킹(묶어준다.)
    - def func(*args) : 여기서 의미하는 * 도 묶음의 의미 !
    - 입력받은 인자가 여러개일 경우 튜플로 묶어서 전달하라는 뜻
```python 
nums = 1,2,3,4,5     # 변수의 수가 맞지 않아 패킹됨 
>>> nums
(1,2,3,4,5)          # 리스트의 경우에도 적용 가능 !

first, *others, last = nums 

>>> first
1
>>> others 
[2,3,4]              # * 를 통한 패킹 - 리스트로 묶임
>>> last
5


def func(n1,n2,*others) :     # 세번째 이후 값들은 튜플로 묶여서 other 자리에 전달됨
    print(type(others))
    print(n1,n2,others,sep='\n')

>>>func(1,2,3,4,5)
1
2
(3,4,5)                       # 이 경우는 리스트 아니고 튜플 ? 
```

     

+ 튜플 언패킹 
    -  '__*__' 가 예외적으로, __함수 호출 시__ ( 정의 x ) 언패킹 용도로 사용됨. 
    - 변수의 형태 및 수와, 튜플 값의 형태 및 수가 일치할 때 자동 언패킹된다.
```python
def func(name,age,h) :
   
    print(name,age,h,sep='\n')


p = ('yoon',13,171)

>>>func(p)
TypeError: func() missing 2 required positional arguments: 'age' and 'h'
# p가 하나의 튜플로 묶여있으니, name 매개변수에만 전달된 상황

>>>func(*p)         # '*' 을 통해 튜플을 언패킹하여 인자로 전달
yoon
13
171 
# 3개의 값으로 언패킹되어 각각 전달되는 것을 확인할 수 있다.
```
```python
p = ('john', (172,68),'usa')
>>> name,height,weight,country = p 
ValueError: not enough values to unpack (expected 4, got 3) # 형태가 일치하지 않기 때문.

>>> name,(height,weight),country = p  # 형태와 수를 완전히 일치시킴.
>>> print(height,country)
172 usa
# 튜플 내의 튜플도 위와 같은 형태로 언패킹
```
### Dict 의 View 객체

+ keys(), values(), items() : 딕셔너리의 __'현재 상태'__ 를 바라보는 View 객체를 생성함
    - view 객체가 생성된 시점이 아닌, view 객체의 iterator가 호출된 시점의 딕셔너리를 참조함 !
```python 
d= dict(a = 1, b = 2, c = 3)
view = dict.items()             # view 객체 생성

for i in view :                 # iterator 호출
    print(i)

d['a'] += 10
d['b'] += 20

for i in view :                 # iterator 호출
    print(i)

                                # view 객체를 새로 생성하지 않아도
                                # iterator 호출 시점에서의 dict 값 반영!
```

+ dict 컴프리헨선
```python
ks = ['a','b','c']
vs = [1,2,3]
d = {k : v for k,v in zip(ks,vs)}

>>> d
{'a': 1, 'b': 2, 'c': 3}
```



### *args , **kargs 

+ 기본적으로 __'*'__ 는 패킹의 용도.
    - 예외적으로 함수 호출 시 (정의 시 x) 는 언패킹의 용도.

+ 함수 정의 시 :
    - def func(*args) : 값들이 튜플로 묶여서 args 에 전달
        - func(1,2,3,4,5)
    - def func(**kargs) : 값들이 딕셔너리로 묶여서 kargs 에 전달
        - func(a = 1 , b = 2)



       
              
+ 함수 호출 시 :
    - func(  *list  )   :   리스트의 값을 풀어서 전달
    - func(  *dict  )   :   딕셔너리의 key 값을 풀어서 전달
    - func( **dict  )   :   딕셔너리를 key = value 형태로 풀어서 전달
    - 딕셔너리를 (Key,Value) 형태로 풀어서 전달하고 싶을때는?
        - func( *dict.items() ) 


## 2020.12.03

### Sequence 타입

+ 저장 순서 정보가 존재하는 자료형 !
    - 변경하지 않는다면 저장한 순서대로 값의 순서가 유지된다.
    - list, tuple, str . . .
    - __인덱싱 / 슬라이싱__ 연산이 가능하다!

### Mapping 타입 

+ 저장된 값의 순서 또는 위치 정보를 기록하지 않는 자료형
    - key 와 value 의 mapping 으로 기록됨 
    - dict
    - 인덱싱 및 슬라이싱 불가능

### Set 타입

+ 수학의 '집합' 을 표현한 자료형 
    - 저장 순서가 없다. ( 내용만 같으면 A == B)
    - set : mutable( 수정 가능 )
    - frozenset : immutable 


## 2020.12.21
### list 정렬


+ sorted(list) : 함수 ( 새로운 객체 )
+ list.sort() : 메소드 ( 리스트 변경 )

```python 
ns = [('Kim', 22),('Lee',25),('Park',27)]

def age(t) :
    return t[1]

ns.sort(key = age) # 정렬 매개변수로 함수 객체를 전달
                   # ns 의 값마다 age를 적용해보고 정렬
print(ns)

ns.sort(key = lambda t : t[0])

print(ns)

ns.sort(key = len)
```
>>> sort 및 sorted 의 key 매개변수

## 2020.12.23

### isinstance, issubclass
```python
class Car :
    def __init__(self,id = None) :
        print("i am car")
        print(id)

class Son(Car) :
    def __init__(self,id) :
        super().__init__(id)
        print("i am son")


s = Son(123)

print(type(list)) # list는 type 클래스의 객체이다
print(isinstance(Son,Car)) # 하위클래스는 인스턴스가 아님!

print(isinstance(list,type)) # list는 type 클래스의 하위클래스가 아닌, 객체이다
print(issubclass(list,type)) # list는 type 클래스의 하위클래스가 아닌, 객체이다
```



### iterable 구현하기 

+ iterable -> __iter__ 메소드 보유 / iter() 에 전달가능  / iterator 를 반환해야 함
+ iterator -> __next__ 메소드 보유


+ iterable & iterator -> __iter__ / __next__ 메소드 둘 다 보유
    - __iter__(self) : return self 
    - 스스로를 반환 / 반환한 self가 __next__ 메소드를 보유하고 있으므로, 
    - iter() 의 결과가 iterator를 반환하는 것 맞음! 즉, iterable !
    - self 는 iter()를 통하여 iterator를 반환하는 iterable 객체이면서,
    - 동시에 __next__ 메소드를 보유한 iterator 임

```python
class Myiterable :
    def __init__(self,data) :
        self.data = data 

    def __iter__(self) :
        return iter(data)

    # 입력된 data의 iterator 를 빌리기 !
    # 아무튼 iter() 로 iterator 를 반환하니까 클래스가 iterable 객체는 맞음.
    # 단, data 에 iterable 객체가 들어와야겠지 ?



class Myiterator :
    def __init__(self,data) :
        self.data = data
        self.count = 0
    def __next__(self)  :
        if self.count >= len(self.data) :
            StopIteration
        self.count += 1
        return self.data[self.count - 1]

    # next()로 하나씩 값을 반환하고, StopIteration 으로 중단하는 
    # 이 클래스는 iterator 객체임
```
>>> iterator만 만들어서 뭐하나. 다른 곳에서 iter()를 통해서 이 iterator 가 반환되게 하려고?

```python
class Myiterator2 :
    def __init__(self,data) :
        self.data = data
        self.count = 0

    def __next__(self)  :      # __next__ 메소드를 보유한 iterator 객체인데,
        if self.count >= len(self.data) :
            StopIteration
        self.count += 1
        return self.data[self.count - 1]

    

    def __iter__(self) :       # __iter__를 통해 스스로를 반환하면 
        return self            # __iter__를 통해 iterator 를 반환하는
                               # iterable의 정의도 만족.


    
```
>>> 스스로가 iterator이자 iterable한 클래스 구현함으로써, iterable한 객체의 본연 용도 사용 가능!


### 연산자 오버로딩 

+ __+__ / __+=__ 가 필요한 상황을 구분하자. 

+ __\__add_\___ : 기존 피연산자의 값을 (되도록) 변경시키지 말기 !
    - ex ) n3 = n1 + n2 
        - 연산자가 n1과 n2의 값을 변경시키지 않음
    
    - 연산의 결과를 새로 return 하는 형식으로 !


+ in-place 연산
    - ex) n1 += n2 처럼, 기존 값 변경되는 경우 가 in-place 연산 
        - n1 = n1 + n2 와 같긴 하나,
        - __+__ 와 __+=__ 가 성격을 달리 해야하는 상황 가정 ! 
    - __\__iadd_\___ 를 정의하여 __+=__ 에 대해 연산자 오버로딩
        - 정의하지 않으면, 원래처럼 풀어서 해석됨!
        - __return self__ 필수 ! 
            - n1 += n2 의 결과로 n1 반환


    - 참고 / immutable 객체 (정수, 문자열) 은 in-place 불가능
        - n = 1
        - n += 1 
        - n = n + 1 
        - n이 새로운 객체 2를 가르키게 됨 ( 가르키던 객체가 변경 불가능 하기 때문 )

## 2020.12.24

### __dict__ / __slots__

+ _\_dict__ : 기본적으로 파이썬의 객체는 딕셔너리를 통해 인스턴스 변수들을 관리.
    - 클래스가 아닌 인스턴스당 하나씩
    - 딕셔너리를 참조하여 변수를 관리함
    - pros) 유연성 확보 : 변수의 추가 및 삭제
    - cons) 효율성 저하 : 내부적으로 관리하는 것 보다 참조하는 과정이 더 생김

+ _\_slots__ : 딕셔너리를 생성하지 않고, 내부에서 변수를 직접 관리
    - 선언된 튜플 외의, 변수의 추가나 삭제를 제한함
        - tuple 은 immutable 객체이기 때문
    

```python
class Point :

    __slots__ = ('x', 'y')       # x, y 좌표 이외의 변수가 추가될 일이 없기에, 제한한다


    def __init__(self, x, y) :
        self.x = x
        self.y = y

p1 = Point(3,5)
p1.w = 1 # 오류 발생 


```


### property 
+ 인스턴스 변수에 직접 접근하지 않고 프로퍼티 객체를 통해 간접 접근
+ 프로퍼티 객체는 클래스 내 변수 형태로 저장
+ P = property(getter,setter)
    - P 라는 변수로 접근하지만, 실은 getter / setter 메소드를 통해 인스턴스 변수에 접근
    - 프로퍼티 객체를 클래스 내 변수 P 에 할당
        - 대입연산자 왼쪽 ( P = ) 형태로 호출 시 : setter / 값 넣기
        - 대입연산자 오른쪽 ( = P ) 형태로 호출 시 : getter / 값 꺼내기

+ P = property() : 객체 생성
+ P = P.getter(getn) 
+ P = P.setter(setn)
    - getter / setter 설정 시 설정된 새로운 객체 반환하기 때문에 P 변수에 다시 할당


```python
class Natural :
    def __init__(self,n):
        self.n =n
    def getn(self) :
        return self.n
    def setn(self, n) :
        self.n = n 
    k = property(getn,setn)

n1 = Natural(1)
n2 = Natural(2)
n3 = Natural(3)


n1.k = n2.k + n3.k  
# n1.setn(n2.getn()+n3.getn())

print(n1.k)
```

```python
class Natural :
    def __init__(self,n):
        self.__n =n

    @property               # == n = property(n)                            
    def n(self) :           # 1. property 객체 생성
        return self.__n     # 2. n 메소드를 getter로 등록
                            # 3. n 이 메소드가 아닌 property를 참조하게 함.
  
  
    @n.setter
    def setn(self, n) :
        self.__n = n 
    k = property(getn,setn)



print(n1.k)
```

## 2020.12.25

### 네스티드 함수 / 클로져

+ 클로져 : 함수의 종료시 사라지는 변수를 잠시 저장해두는 기술

```python
def maker(m):
    def inner(n) :  # 함수도 객체, 함수의 이름은 그 변수명 !
        return m*n
    return inner    # 함수도 객체기에 반환 가능


>>> print(maker)
<function maker at 0x7fa1026b9160>
>>> print(inner)
NameError: name 'inner' is not defined 
# maker 내에서만 존재하는 변수명이기 때문에, 이렇게 호출할 수 없다.


>>> func = maker(2) # inner 라는 함수 객체를 반환하고, func라는 변수명을 붙인 것
>>> print(func)
<function maker.<locals>.inner at 0x7f8355512670>


>>> func(7)     # inner(7) 과 같다는 건데, 
14              # m 은 maker 함수 밖에서 존재하지 않는데
                # n = 입력인자 참조, m = 참조할 대상이 없는데...
                # m * n 이 가능한가..?

# 클로져 기술로 잠시 inner (func) 객체 안에 m 의 값을 저장!

>>> func.__closure__[0].cell_contents
2


```

### 데코레이터

+ 꾸며주고, 보강하는 함수 혹은 클래스 (  + 알파 )

```python
def smile() :
    print("^_^")

def deco(func) :
    def df() :
        print("꾸미기")
        func()
        print("꾸미기")
    return df           # df 객체를 반환하는 deco 함수

>>> smile()
^_^

>>> print(smile)
<function smile at 0x7f92f2fb9160> # smile 변수가 smile 함수를 가리킴





>>> smile = deco(smile)  # deco 는 df 객체를 반환. smile 변수가 df 함수를 가리킴
>>> smile()
꾸미기
^_^
꾸미기


>>> print(smile)
<function deco.<locals>.df at 0x7f92f4513790> # smile 변수가 df 함수를 가리킴


>>> print(smile.__closure__[0].cell_contents) 
<function smile at 0x7f92f2fb9160> # 클로져 내 func 변수가 smile 함수를 가리킴
```
>>> 입력 인자가 없는 함수의 데코레이션  
  
    


```python
def adder_2(n1,n2):
    return n1+n2

def adder_3(n1,n2,n3):
    return n1+n2+n3

def deco(func) :
    def deco_adder(*args) :                 # 입력 인자를 튜플로 패킹 ( 튜플 이름이 args )
        print(*args, sep=' + ', end=' ')    # args 튜플을 언패킹
        print("= {}".format(func(*args)))   # args 튜플을 언패킹해서 func 에 하나씩 전달
    return deco_adder                       # func 는 클로져에 저장된 상태


>>> adder_2 = deco(adder_2)
>>> adder_2(2,3)
2 + 3 = 5

>>> adder_3 = deco(adder_3)
>>> adder_3(2,3,4)
2 + 3 + 4 = 9
```
>>> 전달 인자가 있는 함수의 데코레이션  
  
    


```python


def deco(func) :
    def df() :
        print("꾸미기")
        func()
        print("꾸미기")
    return df           # df 객체를 반환하는 deco 함수


                        # 뒤에 나올 함수가 deco 함수를 통과할 것이라는 뜻.
@deco                   # smile = deco(smile) / 통과한 객체를 다시 smile 변수로 할당
def smile() :
    print("^_^")

```
>>> @ 를 사용하여 간단하게.


## 2020.12.26

### 클래스 변수 / static / 클래스 메소드

+ 클래스 변수 : 객체가 아닌 '클래스 소속'의 변수
    - 모든 객체가 공유함
    - 클래스명으로 직접 호출 / 객체명으로 호출 둘 다 가능
    - 메소드와 같은 라인에 변수 선언

```python
class Simple :
    count = 0       # 클래스 변수의 선언
    def __init__(self) :
        Simple.count +=1 



>>> print(Simple.count)
0
>>> s1 = Simple() 
>>> print(Simple.count)
1

```

+ static 메소드 : 객체가 아닌 '클래스 소속'의 메소드
    - 클래스 변수와 같은 개념
    - 기존 인스턴스 메소드의 전달인자 = self
    - self 는 인스턴스를 메소드의 인자로 전달하는 것이기 떄문에, __self 를 생략한다__
    - staticmethod() / @staticmethod
```python
class Simple :
    count = 0                           # 클래스 변수의 선언
    def __init__(self) :
        Simple.count +=1 
    def get_count() :                   # 인스턴스를 인자로 전달하는 self 생략
        return Simple.count

    get_count = staticmethod(get_count) # static 메소드로 선언

>>> print(Simple.get_count())
0

>>> s1 = Simple() 
>>> print(Simple.get_count())           
1
>>> print(s1.get_count())               # 인스턴스를 통해서도 호출 가능
1
```
```python
class Simple :
    count = 0                           
    def __init__(self) :
        Simple.count +=1 
    @staticmethod             # == get_count = staticmethod(get_count)
    def get_count() :                   \
        return Simple.count


```

+ 클래스 메소드 : static 메소드와 한가지 차이!
    - 인스턴스를 self 키워드를 통해 인자로 전달하였듯,
    - 클래스 자체를 cls 키워드로 인자로 전달 !

```python
class Simple :
    count = 0                           
    def __init__(self) :
        Simple.count +=1 
    @classmethod             
    def get_count(cls) :                   
        return cls.count     # cls == Simple 클래스  


```
>>> static 메소드와 큰 차이가 없어 보임. 언제 클래스 메소드를 사용할까?
```python
class Date :
    def __init__(self,y,m,d) :
        self.year = y
        self.month = m
        self.day = d
        
    def show(self) :
        print('DATE : {}/{}/{}'.format(self.year,self.month,self.day)) 

    @classmethod
    def next_day(cls,d) :                       # Date 객체 d를 입력
        return cls(d.year, d.month, d.day+1)    # d 의 다음날의 Date 객체 반환
    @staticmethod
    def static_next_day(self) :
        return Date


class KDate(Date) :                             # Date 클래스 상속
    def show(self) :
        print('KOREA : {}/{}/{}'.format(self.year,self.month,self.day)) 

class JDate(Date) :                             # Date 클래스 상속
    def show(self) :
        print('JAPAN : {}/{}/{}'.format(self.year,self.month,self.day)) 



>>> day1 = Date(2020,1,1)
>>> day2 = Date.next_day(day1)  # 클래스 메소드 호출
>>> day2.show()
DATE : 2020/1/2                 # 반환된 cls = Date



>>> day1 = KDate(2020,1,1)      
>>> day2 = KDate.next_day(day1) # 클래스 메소드 호출
>>> day2.show()
KOREA : 2020/1/2                # 반환된 cls = KDate



>>> day1 = Date(2020,1,1)       
>>> day2 = JDate.next_day(day1) # 클래스 메소드 호출 / 입력인자 d에 JDate가 아닌 Date 전달
>>> day2.show()         
JAPAN : 2020/1/2                # 반환된 cls = JDate
                                # Date 클래스의 메소드가 호출되었지만,
                                # cls 에 전달된 것은 해당 메소드를 호출한 JDate 임을 볼 수 있다.
```
>>> 클래스 메소드는 인자로 클래스 정보를 받는다. 이정보는 호출 경로에 따라 __유동적__ 이다.


### __name__ / __main__

+ _\_name__ : 파이썬이 자동으로 생성하는 변수
    - 파일명을 가리킨다.
    - 파일별로 각각 _\_name__ 이 생성된다.

+ _\_main__ : 실행의 주체가 되는 스크립트의 name
```python
def pp():
    print('imp 모듈의 pp함수입니다.')
    print('imp.py의 __name__ =', __name__)

def main() :
    print('imp 스크립트 내 main() 이 실행되었습니다.')
    pp()

>>> main()
imp 스크립트 내 main() 이 실행되었습니다.
imp 모듈의 pp함수입니다.
imp.py의 __name__ = __main__

```



```python
import imp

imp 스크립트 내 main() 이 실행되었습니다.
imp 모듈의 pp함수입니다.
imp.py의 __name__ = imp 

# 실행의 주체가 imp 스크립트가 아니기 떄문에, imp 의 __name__ 이 파일명을 참조한다.
```


```python
imp.py

def pp():
    print('imp 모듈의 pp함수입니다.')
    print('imp.py의 __name__ =', __name__)

if __name__ == '__main__' :          
    def main() :
        print('imp 스크립트 내 main() 이 실행되었습니다.')
        pp()

    main()

imp 스크립트 내 main() 이 실행되었습니다.
imp 모듈의 pp함수입니다.
imp.py의 __name__ = __main__
```
>>> _\_name__ 이 _\_main__ 일 때, 즉 실행의 주체일 때만 main() 을 실행한다.

```python
import imp 

>>> imp.pp()
imp 모듈의 pp함수입니다.
imp.py의 __name__ = imp
```
>>> imp의 main() 부는 실행하지 않고, pp() 함수만 가져와서 사용하게 되었다.