# __TIL__ 


## 2020.10.31  




### 힙 (heap)  

+ 모양 : 마지막 레벨 제외 모두 두개의 자식
+ 성질 : 부모노드 > 자식노드 (최대힙)
+ list로 표현하기 : k번째 index의 left, right child == 2k+1 , 2k+2


* heapify_down(k) : A[k]의 자식노드 A[2k+1], A[2k+2] 중 가장 큰 것이 A[k] 자리로 올라가고 A[k]는 밑으로 내려감 !
                    pointer가 k번째 index에 있던 key를 따라간다. pointer, 즉 원래 A[k]의 key가 단말 노드에 도달하거나,
                    자식 노드와 바꾸지 않을때(자식 노드들보다 클 때)까지 while 루프.
                    O(logN)
* make_heap : heap을 list로 표현했을 때 , 마지막 index부터 heap 성질을 만족 ( 부분 heap ) 하도록 heapify_down !
              해당 부분이 heap 성질 만족하면, 앞의 index로 넘어감 ! 
              O(NlogN) or O(N)


* Insert : A.append() +
           A.heapify_up(k) : heapify_down과 반대로, A[k]와 부모노드 A[(k-1)//2]중 큰 것이 올라가고, root에 도달하면 break
           O(logN)


* find_max() : return A[0] -> maxheap의 root 키 값 반환
               O(1)

* delete_max() : root (max값)과 마지막 Index (단말노드) 를 swap 후 단말로 옮겨진 max값을 pop  
                 heapify_down(0)
                 O(logN)


* search는 효율적이지 않기에 지원 X

```python
* heap sort : for k in range(n) :
                   delete_max()

  O(NlogN)
```


### 이진 탐색 트리 (BST) 
+ 이진트리를 Search 에 효율적으로 활용하기 위하여 정의


+ 정의 : 각 노드의 Left subtree 모든 key값은 노드의 key값보다 작아야 하고,  
        각 노드의 right subtree 모든 key값은 노드의 key값보다 커야한다.  

+  _모든 노드에 대하여 위 조건이 만족하면, BST!_


* 이진트리의 순회 (모든 노드 출력하기)  
    
    - Preorder : M - L - R  

    - Inorder : L - M - R  

    - Postorder : L - R - M  

    - 위의 순서로 재귀적으로 순회 !
    

* BST Search 및 Insert 

    + Search :  O(h) 의 탐색 시간 -> h를 작게 유지하기 위해 노력 !


        - find_loc(self,key) : 해당 key 노드를 찾거나, 없다면 들어갈 위치 (key 값의 부모가 될 노드) return

            - while v!= none // self.root 부터 key 값을 비교하며 key가 노드의 key보다 작다면 left, 크다면 right 로 내려감! 

            - v의 parent p도 추적

        

        
    + Insert : O(h) -> Search의 탐색시간 + 상수시간 
        - if find_loc(key) 의 return != key    # key가 노드에 없음 
        - v = node(key) , p=find_loc(key)     # p == none 일 경우 root
        - v.parent = p , p.left(right) = v    # p와 v의 대소 비교
        - self.size += 1  




* BST의 삭제 : m을 찾는 과정 -> O(h)

    + Delete by Merging(x)


        - x를 삭제 후, x의 왼쪽 서브트리 L을 x자리로 올림 
        - 항상 L's key < R's key 이므로, L의 가장 큰 노드 m의 오른쪽으로 R을 병합
        - link 수정

        1. L이 없는 경우 : R이 x자리 대체
        2. x == root 인 경우 : root 값 수정

    
    
    + Delete by Copying(x) 


        - x 노드 자체를 삭제 하는 것이 아니라, L의 가장 큰 노드 m으로 key값만 수정
        - x 자리에 L에서 가장 큰 m이 오면, BST의 조건 L < x < R 만족!
        - 기존 m의 자식 노드들은 m의 부모의 오른쪽 자식이 됨! #m의 자식들은 m이 m의 부모의 오른쪽 자식이었으므로 m의 부모보다 항상 크다!










### 균형이진탐색트리 (Balanced BST)


+ 정의 : 높이 h를 O(logN) 수준으로 유지하도록 조정하는 이진탐색트리를 고안 !

+ 회전 : 조정연산 - 높이
    - Rotate right , left (self,z)

    - z의 왼쪽 서브트리 높이가 더 큰 경우 : right rotation으로 왼쪽 서브트리의 레벨을 끌어 올림! == 전체트리 레벨 끌어올림


    - A < x < B < z < C
    
    - 6개의 링크 수정 !
  
+ 종류
    - AVL  
    - Red-Black
    - 2-3-4
    - Splay 









### AVL    


+ 정의 : 모든 노드에서 각 노드의 Left, Right 서브트리 height 차이가 1 이하인 BST
+ class Node 동일, height라는 멤버 변수 추가
+ class AVL(BST)    # 상속 , Insert 등에서 height 업데이트 필요.


* Insert : 일단 삽입은 똑같이 하나, 높이차가 AVL 조건에 어긋날 때만 추가 조정!  

    - def insert(self,key) :  
        v = super(AVL,self).insert(key)  
          

        #class BST의 insert 호출 
   
    - 삽입된 v 로부터 부모노드 따라 올라가기
      처음으로 높이 차 나는 부모노드 == z
      z, y, x == v에서 z로 가는 경로상의 노드
      
        - Rebalcance (x,y,z) : < - 원래 z 자리에 온 key를 return   # root 값이 바뀐 경우 고려하기 위함

         x, y, z가 linear : 1회, rotate(z)
         x, y, z가 triangle : 2회, rotate(y) , rotate(z) 



* Delete : (부모노드의 밸런스도 깰 수 있음!)
    - DBM or DBC 불러옴    
    - delete(self,u) :
        v = super(AVL,self).DeleteByCopying(u)  

        \# u를 지워서 가장 처음 균형이 깨질’수도’ 있는, 가장 깊은 곳의 노드 v return 해야 함!
        \# v의 부모에서 균형이 깨지는지 , root 까지 관찰
        \# root 가 바뀔 경우에 대비하여, v pointer가 올라가며 자식인 w 기록 
            -> v == none ~ self.root = w

      z == 삭제된 곳에서 올라가며 가장 처음 균형이 꺠진 노드 

      z, y, x ==  z로부터 높이가 __무거운 쪽__ 으로 내려가며 y, x 지정

    - Rebalnce(z,y,x) :
         1회 or 2회 rotate
         무거운 쪽에서 가벼운 쪽으로 나눠주기 


      ### __원래 z자리의 부모 w에서 균형이 깨진다면 ?!__
        - z에서 깨진 균형을 맞추다 보면 그 위의 부모에 영향을 미쳐 높이차가 발생할 수 있음!
        - 계속 부모노드로 올라가며 균형을 맞춘다 !
        - 루트노드까지 모든 level에서, 높이 h 번 로테이트 해야할 수도 있다 !  #  O(logN)번 * O(1)

        - While v!=none   # 루트까지
      
        
          
            

  



  
  
  
  



## 2020.11.03



### Red - Black 트리

+ None 값의 단말노드 / 내부노드
+ 조건 
    1.  모든 노드는 Red / Black
    2.  root 노드 == Black
    3.  단말노드(NIL) == Black
    4.  Red 노드의 자식노드 == Black    # Black 노드의 자식은 상관없음
    5.  각 노드에서 단말노드로 가는 경로상 Black 노드의 수는 항상 같아야 함


* Insert 

    + BST의 insert 연산 호출 : O(logn)

        x.color = red (기본)

    + 4가지 경우    -- 최대 2회 회전, 색 조정 : O(1) 
        1. 처음 insert ( root )
            x.color = black


        2. x.parent.color == black
            
            부모가 black, 단말은 항상 black 
            -> do nothing ( insert를 통해 단말에 삽입되었기 때문 : NiL == black )


        3. x.parent.color == red ,  x.uncle.color == red (부모의 형제)
            x의 부모가 red이므로, x의 형제는 black
            x.grandparent.color == black 임!
            x.grandparent.color = red 으로 조정
            x.parent.color , x.uncle.color = black 으로 조정
            grandparent 입장에서 (parent, uncle)에 black을 준 것.
            경로상 black 의 수(height 아님)는 변하지 않음!
            parent, uncle 입장에서는 공통적으로 증가한 경우이므로 상관없음  

                > grandparent의 부모가 red 라면...?!

                    루트까지 올라가면 앞서 말한대로 루트의 color를 반전하면 되니 간단.물론 그것은 최악의 경우..그건 그렇다치고, 그렇다면 bh에 대한 위반은 없는지?? leaf 노드까지 내려갈 때 거치는 black노드 수는 바뀐것이 없으므로 위반사항이 없다.
                
        4. x.parent.color == red, x.uncle.color == black 

            x-p-g -> linear 
                g 에서 1회 rotate 후 색 재조정
            x-p-g -> triangle
                p에서 1회, g에서 1회 rotate 후 색 재조정

            5번째 조건 -> 단말까지의 경로상의 bh가 같아야함 !

AVL과 Red-Black 모두 search, insert, delete 모두 O(logn)

|회전 수 | AVL | Red-Black |
|---|---|---|
search | - | - 
insert | 2 | 2 
delete | O(logN)  (최악 : 높이 h)| 3 

-> 수행시간은 logN으로 같지만 회전수가 적다!











### 2-3-4 트리 

* 탐색트리(좌측은 작고, 우측은 크다) 이지만, 이진트리는 아님! 
* 자식노드 개수 = 2, 3, 4
* n-노드 : 자식노드가 n개인 노드
* 모든 단말노드가 같은 Level에 존재

* 각 노드의 칸 수 3개 (a,b,c)
    - 2-노드 : a만 존재
    - 3-노드 : a,b
    - 4-노드 : a,b,c
* 1번 자식 < a  < 2번 자식 <  b < 3번 자식 < c < 4번 자식 < d 
* insert 는 항상 단말노드에 !
* insert 할 단말노드가 꽉 찼다면, 단말노드로 가는 경로에 있는 4-노드를 2-노드 2개로 split 하면서 내려감!
* split한 노드의 가운데 값을 부모노드로 올린다. 
  ( 부모노드는 항상 자리가 있다 ! split하지 않고 내려왔으므로, 4-노드가 아니고, a,b,c 가 다 있지 않다.)
* 모두 split 했는데도 단말노드가 꽉 차있다면 단말노드를 split하고 가운데 값을 위로 올림


+ delete : 최악 : level별로 : O(logN)
    - 루트 - 단말로 가는 경로 상 2-노드를 3 or 4-노드로 바꾸기 
    1. 2-노드의 형제 중 3, 4 노드가 있는 경우 : 부모와 함께 rotate 느낌으로 !

    2. 2=노드의 형제가 모두 2=노드인 경우 : 부모의 한쪽과 합쳐서 3-노드 만들기 (fusion)

    3. 루트가 2-노드인 경우 : 새로운 루트가 자식 2개를 루트와 합쳐 만들어짐
  








### Red-Black 트리와 2-3-4 트리 간 치환


+ 2-노드 : black 트리로 치환
+ 3-노드 : 2 Level에 걸쳐 큰 key부터 black-red 순으로
+ 4-노드 : 가운데 있는 key - black , 양쪽 key - red, 양쪽 자식 노드로 




### Union-Find 

+ height가 작은 쪽에서 큰 쪽으로 union
+ height가 같은 경우는 상관 없지만, union 후 height += 1

+ rank (height)가 1 증가할때 노드 수가 두배 증가한다 ???? 높이는 같지만 수평으로 노드가 많을 수도 있잖아!
+ 아 ! Nh는 '최소개수' 기준 일때 두배씩! Nh = 2^h (Nh = 전체 노드 n개, 높이 h인 경우 최소노드개수 )
+  h<=log2N
+ find , union -> O(h) = O(log2N)





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
b=a
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
```

## 2020.11.19

### Iterator, Generator, yield

+ iterator : 한 번에 하나씩 그 객체의 elements에 순서대로 액세스 할 수 있는 객체

    + 'iterable' 한 객체가 iter() 함수를 통하여 'iterator 객체'로 생성됨
    + iter(iterable) -> iterator

    - iter() , next() 지원
    + iter(f) 가 정의된 f.\_\_iter__() 메소드를 호출 : next 메소드를 가지는 iterator 객체를 반환
    + next(f) 가 정의된 f.\_\_next__() 메소드를 호출 : 반복자를 입력받아 다음 요소 반환
    
+ generator 는 iterator의 특수한 형태 
    + generator : yield 를 통하여 next가 호출될 때 마다 다음 값 반환 
    + yield 명령어로 iterator를 만드는 함수

+ return : 값을 함수 외부로 전달하고, 함수를 종료
+ yield : 값만 함수 외부로 전달하고, 함수를 종료하지 않음 (함수의 일시정지)
    + 다음 next()가 호출될 때까지 대기 ?  
      
       
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



>>> list와 set 같은 집합에 대한 iterator는 이미 모든 값들을 저장해 둔 상태이지만, generator는 모든 값들을 갖지 않은 상태(미정)에서 yield에 의해 하나씩만 데이터를 만들어 가져온다는 차이점이 있다.




iter는 반복을 끝낼 값을 지정하면 특정 값이 나올 때 반복을 끝냅니다. 이 경우에는 반복 가능한 객체 대신 호출 가능한 객체(callable)를 넣어줍니다. 참고로 반복을 끝낼 값은 sentinel이라고 부르는데 감시병이라는 뜻입니다. 즉, 반복을 감시하다가 특정 값이 나오면 반복을 끝낸다고 해서 sentinel입니다.

+ iter(호출가능한객체, 반복을끝낼값)
예를 들어 random.randint(0, 5)와 같이 0부터 5까지 무작위로 숫자를 생성할 때 2가 나오면 반복을 끝내도록 만들 수 있습니다. 이때 호출 가능한 객체를 넣어야 하므로 매개변수가 없는 함수 또는 람다 표현식으로 만들어줍니다.
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



