# __TIL__ 

  

## 2020.10.31

  
### 힙 (heap)

'''
+ 모양 : 마지막 레벨 제외 모두 두개의 자식
+ 성질 : 부모노드 > 자식노드 (최대힙)
'''

### 이진 탐색 트리 (BST)

'''
+ 정의 : 각 노드의 Left subtree 모든 key값은 노드의 key값보다 작아야 하고,  
        각 노드의 right subtree 모든 key값은 노드의 key값보다 커야한다.  
+  _모든 노드에 대하여 위 조건이 만족하면, BST!_


* BST의 순회 (모든 노드 출력하기)  
    
    - Preorder : M - L - R  

    - Inorder : L - M - R  

    - Postorder : L - R - M  

    - 위의 순서로 재귀적으로 순회 !
    
* BST 탐색 및 삽입 

    - find_loc(self,key) : 해당 key 노드를 찾거나, 없다면 들어갈 위치 반환
        - self.root 부터 key 값을 비교하며 left , right 로 내려감!
        - 반환 값을 이용하여 Insert !
    
* BST의 삭제

    - Delete by Merging

    - Delete by Copying
'''

### 균형이진탐색트리

'''
+ 정의 : 높이 h를 O(logn)수준으로 유지하는 이진탐색트리 !

+ 회전
    - Rotate right , left (self,z)
        6개의 링크 수정 !
'''
### AVL     
'''
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
'''

## 2020.11.03

### Red - Black 트리
'''
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

|회전 수 | AVL | Red-Black   |
|---|---|---|
search | - | - 
insert | 2 | 2 
delete | O(logN)  (최악 : 높이 h)| 3 

-> 시간은 logN으로 같지만 회전수가 적다!
'''
### 2-3-4 트리 
'''
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
'''
### Red-Black 트리와 2-3-4 트리 간 치환
'''
+ 2-노드 : black 트리로 치환
+ 3-노드 : 2 Level에 걸쳐 큰 key부터 black-red 순으로
+ 4-노드 : 가운데 있는 key - black , 양쪽 key - red, 양쪽 자식 노드로 
'''