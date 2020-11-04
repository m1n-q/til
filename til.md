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
        - 기존 m의 자식 노드들은 m의 부모의 오른쪽 자식이 됨! # m은 부모의 오른쪽 자식이었고, m의 자식들은 m의 부모보다 크다!










### 균형이진탐색트리


+ 정의 : 높이 h를 O(logn)수준으로 유지하는 이진탐색트리 !

+ 회전
    - Rotate right , left (self,z)
        6개의 링크 수정 !
  












### AVL    


+ 정의 : 모든 노드에서 1이하의 height 차이 유지 

+ AVL 삽입
    node에 height 인스턴스변수 필요
    삽인된 v 로부터  부모노드 따라올라가기
    처음으로 높이 차 나는 부모노드 - z
    Z y x <-  v에서 z로 가는 경로상 노드
    Rebalcance (xyz)  < - 원래 z 자리 리턴

    xyz가 1자일경우 1회 로테이트
    삼각형일경우 2회 로테이트( y에서  1회 , z에서 1회)

+ Avl 삭제 (부모노드의 밸런스도 깰 수 있음!)
    Dbm or dbc 불러옴 ( 처음 균형이 깨질’수도’ 있는 노드 반환)

    (가장 처음 균형이꺠진 노드 z)

    균형 깨진 z로부터 높이가 무거운쪽으로 내려가면서 z,y,x 

    무거운쪽에서 가벼운쪽으로 나눠주기 
    Z 로테이트…

    원래 z자리의 부모 w에서 균형이 깨진다면 ?!

    — 루트노드까지 올라가는 높이 h 번의 로테이트 될수도 !

    While v!=none (루트)  
      
        
          
            

  



  
  
  
  



## 2020.11.03






### Red - Black 트리

+ 정의 :


+ 삽입
    + BST의 insert 연산 호출 :O(logn)

        x.color = red (기본)
    + 4가지 경우    -- 최대 2회 회전, 색 조정 : O(1) 
        - 처음 insert
            x.color = black


        - x.parent.color == black
            부모가 black, 단말은 항상 black 
            -> do nothing


        - x.parent.color == red
            x의 부모가 red이므로, x의 형제는 black

            - x.uncle.color == red (부모의 형제)
                x.grandparent.color == black 임!
                x.grandparent.color = red 으로 조정
                x.parent.color , x.uncle.color = black 으로 조정
                grandparent 입장에서 (parent, uncle)에 black을 준 것.
                경로상 black 의 수(height 아님)는 변하지 않음!
                parent, uncle 입장에서는 공통적으로 증가한 경우이므로 상관없음   

                
            - x.uncle.color == black (부모의 형제)

                x-p-g -> linear 
                    p에서 1회 rotate 후 색 재조정
                x-p-g -> triangle
                    x에서부터 1회, p에서 1회 rotate 후 색 재조정

                5번째 조건 -> 단말까지의 경로상의 bh가 같아야함 !

AVL과 Red-Black 모두 search, insert, delete 모두 O(logn)

|회전 수 | AVL | Red-Black |
|---|---|---|
search | - | - 
insert | 2 | 2 
delete | O(logN)  (최악 : 높이 h)| 3 

-> 시간은 logN으로 같지만 회전수가 적다!











### 2-3-4 트리 

* 탐색트리(좌측은 작고, 우측은 크다) 이지만, 이진트리는 아님! 
* 자식노드 개수 = 2, 3, 4
* n-노드 : 자식노드가 n개인 노드
* 모든 단말노드가 같은 Level에 존재

* 각 노드의 칸 수 3개 (a,b,c)
    - 2-노드 : a만 존재
    - 3-노드 : a,b
    - 4-노드 : a,b,c
* 1번 자식 < a __|__ a < 2번 자식 < b  __|__ b < 3번 자식 < c __|__ c < 4번 자식 < d 
* insert 는 항상 단말노드에 !
* insert 할 단말노드가 꽉 찼다면, 단말노드로 가는 경로에 있는 4-노드를 2-노드 2개로 split 하면서 내려감!
* split한 노드의 가운데 값을 부모노드로 올린다. ( 4-노드는 항상 꽉 차있고 , 4-노드가 아니라면 부모노드가 항상 자리가 있음)
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
+ 아 ! Nh는 최소개수 기준 일때 두배씩! Nh = 2^h (Nh = 전체 노드 n개, 높이 h인 경우 최소노드개수 )
+  h<=log2N
+ find , union -> O(h) = O(log2N)