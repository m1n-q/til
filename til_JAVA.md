## 2020.11.24
### JAVA

### static (클래스, 인스턴스 변수) : 
+ 변수나 메소드의 소속을 인스턴스가 아닌 클래스 소속으로 바꿔줌!
+ 객체 생성 없이 호출 가능


+ 클래스 메소드는 그 메소드의 작업을 수행하는데 필요한 값들을 모두 매개변수로 받아서 처리하고,
+ 인스턴스 메소드는 작업을 수행하는데 인스턴스 변수를 필요로 하는 메소드이다.(즉, 인스턴스 변수와 관련된 작업을 하는 것)

### final 
+ final이 적용된 변수의 값은 초기화 된 이후 변경 불가
+ final이 적용된 메소드는 서브클래스에서 오버라이딩 불가
+ final이 적용된 클래스는 하위클래스 생성 불가


### 추상 클래스 (abstract)
+ 일반 클래스 및 인스턴스 생성불가
+ 상속을 통해 서브클래스에서 인스턴스 생성해야함
+ 추상 클래스에는 필드, 일반메소드, 추상메소드 모두 존재할 수 있다. __단지 인스턴스 생성을 할 수 없을 뿐!__

```java

abstract class Car {
    int speed = 0;
    String color ;

    void upSpeed(int speed) {
        this.speed += speed;
    }
}

class Sedan extends Car {

}
```

### 추상 메소드 (코드가 없는 껍데기 메소드)

+ 서브클래스에서 __오버라이딩__ 하기 위해 사용
+ 슈퍼클래스에서는 추상 메소드로 껍데기만 만들어두고 , 실제 내용은 __서브클래스에서 채워넣음__
+ __abstract 반환형 메소드이름(파라미터);__
```java

abstract class Car {
    int speed = 0;
    String color ;

    void upSpeed(int speed) {
        this.speed += speed;
    }
    abstract void work();           // 추상메소드를 가지려면 반드시 추상클래스여야함
}
class Sedan extends Car {
    void work() {
        System.out.println("승용차가 사람을 태우고 있습니다.");
    }
}
class Sedan extends Car {
    void work() {                   // 서브클래스에서는 반드시 추상메소드를 오버라이딩 해야한다.
        System.out.println("승용차가 사람을 태우고 있습니다."); 
    }
}

class Truck extends Car {
    void work() {
        System.out.println("트럭이 짐을 싣고 있습니다.");
    }
}

    

```


### 인터페이스
+ __직접 인스턴스 생성 불가.__
+ 추상클래스와 차이점 :
+ __필드 & 추상메소드 가질 수 있으나 , 일반메소드 & 생성자 가질 수 없음__
+ __필드값도 static final , 즉 상수화한 필드만 사용할 수 있으며, 반드시 초기화해야함__
+ 상속 : __implements__
+ 오버라이딩 : 인터페이스의 추상메소드를 완성할 때는 __public__ 키워드를 붙인다
+ __다중 상속을 위해 필요 !__
    - class 탱크 extends 자동차, 대포  : JAVA에서 지원하지 않음!
    - class 탱크 implements 자동차, 대포 : 인터페이스를 통한 다중 상속으로 구현!
```java 
interface Car {
    static final int CAR_COUNT = 0; // 필드 정의
    abstract void work(); // 추상메소드

}
class Sedan implements Car {
    public void work(){                // 오버라이딩 : 인터페이스의 추상메소드를 완성할 때는 public 키워드를 붙인다
        System.out.println("승용차가 사람을 태우고 있습니다."); 
    }
}
class Truck implements Car {
    public void work(){                // 오버라이딩 : 인터페이스의 추상메소드를 완성할 때는 public 키워드를 붙인다
        System.out.println("트럭이 짐을 싣고 있습니다.");
    }
}
```