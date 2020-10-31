# __TIL__ 

  

## 2020.10.31

  
### 힙 (heap)


    + 모양 : 마지막 레벨 제외 모두 두개의 자식
    + 성질 : 부모노드 > 자식노드 (최대힙)


### 이진 탐색 트리 (BST)


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
        
    ### Red - Black 트리
