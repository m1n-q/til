# TIL_자료구조

## 2020.10.31  




### 힙 (heap)  

+ 모양 : 마지막 레벨 제외 모두 두개의 자식
+ 성질 : 부모노드 > 자식노드 (최대힙)
+ list로 표현하기 : k번째 index의 left, right child == 2k+1 , 2k+2


* heapify_down(k) : A[k]의 자식노드 A[2k+1], A[2k+2] 중 가장 큰 것이 A[k] 자리로 올라가고 A[k]는 밑으로 내려감 !  

                    pointer가 k번째 index에 있던 key를 따라간다.   
                    pointer, 즉 원래 A[k]의 key가 단말 노드에 도달하거나,
                    자식 노드와 바꾸지 않을때(자식 노드들보다 클 때)까지 while 루프.  
                      
                      
                    
                    즉, k 자리에서 부분 heap을 만들고 k를 밑으로 내려보냄, 또 그 새로운 k 자리에서 heap 조건 맞추며 단말까지!



                    근데 만약 A[k]와 두 child는 작은 heap을 만족하는데,  
                    child와 child의 자식들이 heap을 만족하지 않는다면?  
                    while문의 조건에 의하면 A[k]와 두 child 중 A[k]값이 가장 크면 더 내려가지 않는데 ?

                    -> make_heap 연산에서 마지막 인덱스부터 heapify_down을 하므로,  
                       위와 같은 상황이 올 수 없음 ! 이미 밑에 것부터 작은 삼각형을 만들며 올라왔기 때문!  
                       따라서 make_heap에서는 윗 레벨이 작은 heap을 만족하여  
                       밑 레벨을 체크해주지 못하는 경우가 없음!               


                    -> 위쪽 레벨의 노드가 내려오며 밑에서 만들어진 삼각형을 망가뜨리는 경우에는 또  
                       while문의 내용처럼 k 인덱스가 내려가며 재조정 해줌!  
                            
    
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


+ 정의 : 각 노드의 __Left subtree__ 모든 key값은 노드의 key값보다 작아야 하고,  
        각 노드의 __right subtree__ 모든 key값은 노드의 key값보다 커야한다.  

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

## 2020.11.20

### Hash Table



+ collision 회피 방법 
    1. Open Addressing : linear probing, quadratic probing, double hashing
        - cluster size 에 연산의 속도, 비교회수가 비례한다.
        - m>= 2n , 즉, 50%  이상의 빈 슬롯을 유지해주며 m 을 키워준다면 + C-Universial hash function 사용해야함.
        - cluster size 가 평균 O(1) 수준으로 유지될 수 있따! 즉, set, search, remove 연산이 O(1) 시간에 가능해진다! 


    2. Chaining : 각 slot에 한방향 연결리스트 만듦!
        - set(key) : push_front(key), 즉 O(1) 시간
        - search(key) : O(연결리스트의 길이 == 충돌 key의 평균 개수)
        - remove(key) : O(연결리스트의 길이 == 충돌 key의 평균 개수)
        - C-Universial hash function을 이용하면, 연결리스트 길이가 O(1) 수준 유지 가능 



#### Open Addressing - linear probing
_____
#### pseudo code
```python

### remove ###

# k에서 출발했을 때 순서가 k -> i -> j 여야지! 
# 인덱스 숫자의 크기로 보면  
1. k < i < j 
2. j < k < i
3. i < j < k 

다 가능한 경우인거지 !
j값 탐색의 출발을 k== 원래 해쉬펑션(key) 에서부터 탐색한다는 것!


i = find_slot(key) 삭제할 키의 인덱스를 찾음 / if 없으면 None 리턴

-> i 자리가 비워짐 ! 
-> i 를 메꿀 j를 찾아야함
-> 일단 j = i 로 하고 출발 ! (비워진 칸에서 출발) # H[i] == 빈 슬롯 # H[j] == 빈 곳으로 이사 보내야할 슬롯 찾는중
-> H[i] = None
while True :
    H[i] = None # key 값 가진 H[i] 자리 비웠음
    while True : # H[j] 찾기
        j = (j+1)%m # 내려가며 j 찾기
        if H[j] is unoccupied : # 빈 칸까지 내려갔는데 J가 없다? 이사시킬 넘이 없다 ! key 삭제만 하고 끝났음
             return key    #return으로 안과 밖의 while 루프를 끝냄   
        k = f(H[j].key) # 원래 j에 있는 Key 가 있어야할 자리 k
        if (k<i<=j) : # H[j] 를 H[i]로 옮겨야하는 j 찾음!
            break  # 안쪽 while만 끝냄
    H[i] = H[j] # H[i]로 이사시켰음 !, j번째 슬롯이 비워져야지! 
    i = j # 위로 올라가서 H[i==j] = None으로 j번째 슬롯을 비우는거임 ! (비워진 j 에 대하여 또 반복)
```




#### 구현
```python
class Node :
    def __init__(self,key) :
        self.key=key 
    def __str__(self) :
        return str(self.key)



class Hash_table :

    def __init__(self, size=0) :
        
        self.size = size
        self.data = [None] * size
    
    def showdata(self) :
        for i, j in enumerate(self.data) :
            print( f'index{i} => {j}' )



    def hashfunc(self, key) :
        if key == None :
            return None
        else :
            return key % self.size 
    
    def find_slot(self,key) :
        k = self.hashfunc(key)  
        H = self.data
        m = self.size
        while H[k] != None and H[k] != key : # 빈칸이거나, 찾는 key를 만나면 멈춤. 
                                             # 빈칸을 만나서 멈췄는데, 찾는키가 빈칸 뒤에 있다면..?! ( 이런경우는 중간에서 지워진 경우뿐 ) 
                                             # remove 함수에서 빈칸을 안만들도록 설계하기 !
            k = (k+1)%m
            if k == self.hashfunc(key) :
                return 'FULL'
        else :
            return k    # 빈칸이면 들어갈 자리, 찾는 key라면 위치를 나타냄
            

    def setkey(self,key) :
        H = self.data
        i = self.find_slot(key) 
        if i == 'FULL' :    # 꽉 차있으면 None 리턴 /// size를 키워줘야함 .........!
            print(i)
            return None
        elif H[i] == None :     # find_slot 한 i가 빈 칸이면 삽입
            H[i] = key
            print(f'key {key} is inserted in index[{i}]')
        else :                  # find_slot 한 i가 이미 있는 값이라고 안내
            print(f'key {key} is already in index[{i}]')
        return key

    
    def remove(self, key) :
        H = self.data
        i = self.find_slot(key) # remove로 비워진 자리 i
        
        if H[i] == None : # 이미 있는걸 찾은게 아니면 들어갈 자리를 찾은 것. 테이블에 key 가 아직 없다는 것!
            return None
        
        j = i #i에서부터 내려가기 ( 어차피 i 위의 값은 i를 못채움 )
        while True : # 비워진 H[i]에서부터 출발하여, 그 빈 자리를 채울 j값을 찾아감 
            H[i] = None # H[i] 를 비움
            
            while True :    # j 찾기
                j = (j+1)%self.size
                print(j)
                k = self.hashfunc(H[j]) # j가 원래 들어갈 자리 k
                if H[j] == None : # ---- (2)
                    return key
                if k <= i <= j :#or (j < k < i) or ( i < j < k ) :
                    break           # 올릴 j 값을 찾았으니, 안쪽 while 루프를 빠져나감 ---- (1)

            H[i] = H[j]
            i = j # H[j] 가 비워졌으니, j를 i로 설정해서 바깥쪽 while에서 또 비우고 
                  # 새로운 빈칸 H[i] 에 대하여 같은 작업 반복해줘야 함.......!
                  
                  
                  
                  
                  
                  # 그럼 언제 멈추냐 ? 더이상 올릴 i 로 올릴 j가 없을때 ! 
                  # 즉, 클러스터의 끝까지 갔는데 j가 없을때 !
                  # 즉, j가 빈칸을 만나게 될때 !   # ---- (2)


                  # 빈 칸을 만났는데 , 뒤에 (1) 조건을 만족하는 j가 있으면 어떡하냐 ?
                  # 애초에 setkey 함수에 따라 삽입 됐다면 클러스터에 빈 칸을 두고 밀리는 경우가 불가능 // 
                  # 사이에 빈 칸이 생기는건 remove된 경우만 가능한데, 지금 빈칸을 만들지 않는 remove를 정의하는 중임! 
                  # 그러니까 빈 칸을 만났을때 (i에서 시작하는 클러스터의 끝에 갔을때)
                  # 더이상 i로 올릴 j가 없다고 끝내는 remove를 정의하면 i와 j 사이에 빈칸이 있을 수 없음.
                  # i 밑 클러스터의 끝까지 j를 찾지 못하고 빈 칸을 만나면, 더이상 i로 올릴 수 있는 값이 없다고 while루프를 끝내도 무방.
                  # 더이상 추가적인 이동이 발생하지 않으므로, 새로운 빈칸이 생기지 않음 -> 바깥쪽 while루프까지 종료,


ht= Hash_table(10)
ht.setkey(2)
ht.setkey(3)
ht.setkey(4)
ht.setkey(12)
ht.setkey(15)
ht.setkey(16)
ht.showdata()
ht.remove(3)
ht.showdata()


```