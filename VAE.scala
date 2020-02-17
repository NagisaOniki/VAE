//report4
//変分自己符号化器

package report4

import breeze.linalg._
import CLASS._


/////////////////////////VAEplot(z=2)///////////////////////////////////////
object VAEplot{

  def main(){

    val dn = 100 // 学習データ数 ★

    val rand = new scala.util.Random(0)

    //-----------ネットワーク構成------------------
    //encode
    val Aen_1 = new Affine(784,100)
    val Ten_2 = new Tanh()
    val Aen_3 = new Affine(100,100)
    val Ten_4 = new Tanh()

    val A_myu = new Affine(100,2)
    val A_sig = new Affine(100,2)

    //decode
    //ネットワーク構成A
    /*val Ade_1 = new Affine(2,100)
    val Tde_2 = new Tanh()
    val Ade_3 = new Affine(100,100)
    val Tde_4 = new Tanh()
    val Ade_5 = new Affine(100,784)
    val Sde_6 = new Sigmoid()
     */

    //decode
    //ネットワーク構成B
    val Ade_1 = new Affine(2,100)
    val Tde_2 = new Tanh()
    val Ade_3 = new Affine(100,784)
    val Sde_4 = new Sigmoid()

    //A
    /*Aen_1.load("saveA0-Aen_1")
    Aen_3.load("saveA0-Aen_3")
    A_myu.load("saveA0-A_myu")
    A_sig.load("saveA0-A_sig")
    Ade_1.load("saveA0-Ade_1")
    Ade_3.load("saveA0-Ade_3")
    Ade_5.load("saveA0-Ade_5")
     */

    //B
    Aen_1.load("saveB0-Aen_1")
    Aen_3.load("saveB0-Aen_3")
    A_myu.load("saveB0-A_myu")
    A_sig.load("saveB0-A_sig")
    Ade_1.load("saveB0-Ade_1")
    Ade_3.load("saveB0-Ade_3")
  

    //-------データの読み込み----------
    val (dtrain,dtest) = VAE.load_mnist("/home/share/number")

    for((x,n)<-dtrain.take(dn)) { //x:入力、n:正解
      //----------forward---------------
      val y0 = Ten_4.forward(Aen_3.forward(Ten_2.forward(Aen_1.forward(x))))
      val y_myu = A_myu.forward(y0)
      val y_sig = A_sig.forward(y0)
      val y_sigep = A_sig.forward(y0).map(_*rand.nextGaussian * 0.1)

      val z = y_myu.zip(y_sigep).map{case(a,b)=>a+b}
     
      println(n+","+z(0)+","+z(1))

    }
  }
}


///////////////////VAE(-2,-2)~(2,2)/(z=2)///////////////////////////////
object VAEarea{

  def main(){

    val rand = new scala.util.Random(0)

    //-----------ネットワーク構成------------------
    //encode
    val Aen_1 = new Affine(784,100)
    val Ten_2 = new Tanh()
    val Aen_3 = new Affine(100,100)
    val Ten_4 = new Tanh()

    val A_myu = new Affine(100,2)
    val A_sig = new Affine(100,2)

    //decode
    //ネットワーク構成A
  /*  val Ade_1 = new Affine(2,100)
    val Tde_2 = new Tanh()
    val Ade_3 = new Affine(100,100)
    val Tde_4 = new Tanh()
    val Ade_5 = new Affine(100,784)
    val Sde_6 = new Sigmoid()
*/
     

    //decode
    //ネットワーク構成B
    val Ade_1 = new Affine(2,100)
    val Tde_2 = new Tanh()
    val Ade_3 = new Affine(100,784)
    val Sde_4 = new Sigmoid()
     

    //A
   /* Aen_1.load("saveA0-Aen_1")
    Aen_3.load("saveA0-Aen_3")
    A_myu.load("saveA0-A_myu")
    A_sig.load("saveA0-A_sig")
    Ade_1.load("saveA0-Ade_1")
    Ade_3.load("saveA0-Ade_3")
    Ade_5.load("saveA0-Ade_5")
 */    

    //B
    Aen_1.load("saveB0-Aen_1")
    Aen_3.load("saveB0-Aen_3")
    A_myu.load("saveB0-A_myu")
    A_sig.load("saveB0-A_sig")
    Ade_1.load("saveB0-Ade_1")
    Ade_3.load("saveB0-Ade_3")
  

    //-------データの読み込み----------
    val (dtrain,dtest) = VAE.load_mnist("/home/share/number")
    //----------画像データをまとめる--------
    var ds = (0 until 441).map(i=>dtrain(i)._1).toArray

    var z = new Array[Double](2)

    var j = 0//画像更新
   
    var y = 2.0
    for(p<-0 until 21){
      z(1) = y
      y += -0.2

      var x = 2.0
      for(q<-0 until 21){
        z(0) = x
        x += -0.2
 
        //A
        //val y1 = Sde_6.forward(Ade_5.forward(Tde_4.forward(Ade_3.forward(Tde_2.forward(Ade_1.forward(z))))))
        //B
        val y1 = Sde_4.forward(Ade_3.forward(Tde_2.forward(Ade_1.forward(z))))

        //-----入力画像更新--------
        ds(j)=y1
        j += 1
      }
    }
    //------------画像作成--------------------
    Image.write(f"VAE-areaB.png" , VAE.make_image(ds,21,21))


  }//main
}//VAEarea



/////////////////////////VAElinear(z=20)/////////////////////////////////
object VAElinear{

  def main(){

    val dn = 100 // 学習データ数 ★
    val rand = new scala.util.Random(0)

    //-----------ネットワーク構成------------------
    //encode
    val Aen_1 = new Affine(784,100)
    val Ten_2 = new Tanh()
    val Aen_3 = new Affine(100,100)
    val Ten_4 = new Tanh()

    val A_myu = new Affine(100,20)
    val A_sig = new Affine(100,20)

    //decode
    //ネットワーク構成A
  /*  val Ade_1 = new Affine(20,100)
    val Tde_2 = new Tanh()
    val Ade_3 = new Affine(100,100)
    val Tde_4 = new Tanh()
    val Ade_5 = new Affine(100,784)
    val Sde_6 = new Sigmoid()
*/
    //decode
    //ネットワーク構成B
    val Ade_1 = new Affine(20,100)
    val Tde_2 = new Tanh()
    val Ade_3 = new Affine(100,784)
    val Sde_4 = new Sigmoid()

    //A
  /*  Aen_1.load("saveA1-Aen_1")
    Aen_3.load("saveA1-Aen_3")
    A_myu.load("saveA1-A_myu")
    A_sig.load("saveA1-A_sig")
    Ade_1.load("saveA1-Ade_1")
    Ade_3.load("saveA1-Ade_3")
    Ade_5.load("saveA1-Ade_5")
*/
    //B
    Aen_1.load("saveB1-Aen_1")
    Aen_3.load("saveB1-Aen_3")
    A_myu.load("saveB1-A_myu")
    A_sig.load("saveB1-A_sig")
    Ade_1.load("saveB1-Ade_1")
    Ade_3.load("saveB1-Ade_3")
  

    //-------データの読み込み----------
    val (dtrain,dtest) = VAE.load_mnist("/home/share/number")

    //----------画像データをまとめる--------
    var ds = (0 until 100).map(i=>dtrain(i)._1).toArray
    
    val x1 = dtrain(1)._1 //画像
    val x2 = dtrain(2)._1

    //-----------encode(x1)----------------
    val y0_x1 = Ten_4.forward(Aen_3.forward(Ten_2.forward(Aen_1.forward(x1))))
    val y_myu_x1 = A_myu.forward(y0_x1)
    val y_sig_x1 = A_sig.forward(y0_x1)
    val y_sigep_x1 = A_sig.forward(y0_x1).map(_*rand.nextGaussian * 0.1)
    val z1 = y_myu_x1.zip(y_sigep_x1).map{case(a,b)=>a+b}

    //-----------encode(x2)--------------------
    val y0_x2 = Ten_4.forward(Aen_3.forward(Ten_2.forward(Aen_1.forward(x2))))
    val y_myu_x2 = A_myu.forward(y0_x2)
    val y_sig_x2 = A_sig.forward(y0_x2)
    val y_sigep_x2 = A_sig.forward(y0_x2).map(_*rand.nextGaussian * 0.1)
    val z2 = y_myu_x2.zip(y_sigep_x2).map{case(a,b)=>a+b}

    //-----------decode---------------
    var t = 0d
    var j = 0
    for(i<-0 until 100){
      val z = z1.map(_*t).zip(z2.map(_*(1-t))).map{case(a,b)=>a+b}
      t += 0.01
      //A
      //val y1 = Sde_6.forward(Ade_5.forward(Tde_4.forward(Ade_3.forward(Tde_2.forward(Ade_1.forward(z))))))
      //B
      val y1 = Sde_4.forward(Ade_3.forward(Tde_2.forward(Ade_1.forward(z))))

      //-----入力画像更新--------
      ds(j)=y1
      j += 1

    }

    //------------画像作成--------------------
    Image.write(f"VAE-linearB.png" , VAE.make_image(ds,10,10))

  }

}




/////////////////////////VAE//////////////////////////////////
object VAE{

  //-----------実験データ取得--------------
  def load_mnist(dir:String) = {
    def fd(line:String) = line.split(",").map(_.toDouble / 256).toArray
    def ft(line:String) = line.split(",").map(_.toInt).toArray
    val train_d = scala.io.Source.fromFile(dir + "/train-d.txt").getLines.map(fd).toArray
    val train_t = scala.io.Source.fromFile(dir + "/train-t.txt").getLines.map(ft).toArray.head
    val test_d = scala.io.Source.fromFile(dir + "/test-d.txt").getLines.map(fd).toArray
    val test_t = scala.io.Source.fromFile(dir + "/test-t.txt").getLines.map(ft).toArray.head
    (train_d.zip(train_t), test_d.zip(test_t))
  }

  //-----------クロスエントロピー-------------------
  def CE(a:Array[Double] , b:Array[Double])={
    var sum = 0d
    for(i<-0 until a.size){
      sum += -(a(i) * math.log(b(i)+1e-8) + ( (1-a(i)) * math.log(1-b(i)+1e-8) ) )
    }
    sum
  }

  //-----------平均二乗誤差----------------
    //a:出力、b:正解
    def MSE(a:Array[Double] , b:Array[Double])={
      var sum = 0d
      for(i<-0 until a.size){
        sum += (a(i) - b(i)) * (a(i) - b(i))
      }
      sum
    }

  //-------------------出力を画像に変換----------------------------
  def make_image(xs:Array[Array[Double]] ,H:Int ,W:Int)={
    val im = Array.ofDim[Int]( H * 28 , W * 28, 3 ) //hight,width,RGB
    for(i<-0 until H; j<-0 until W){
      for(p<-0 until 28; q<-0 until 28; r<-0 until 3){
        im(i*28+p)(j*28+q)(r) = (xs(i*W+j)(p*28 + q)*255).toInt
      }
    }
    im
  }

  //-----------main---------------------
  def main(args:Array[String]){

    val ln = 1000 // 学習回数 ★
    val dn = 100 // 学習データ数 ★
    val tn = 100 // テストデータ数 ★

    val rand = new scala.util.Random(0)


    //-------データの読み込み----------
    val (dtrain,dtest) = load_mnist("/home/share/number")

    //----------画像データをまとめる--------
    var ds = (0 until 100).map(i=>dtrain(i)._1).toArray
    Image.write(f"VAE-original.png" , make_image(ds,10,10))

    //------------ヒストグラム作成-----------------
    var histArray1 = new Array[Double](100)
    for(i<-0 until 100){
      histArray1(i) = ds(i)(13+14*28)
    }


    //-----------ネットワーク構成------------------
    //encode
    val Aen_1 = new Affine(784,100)
    val Ten_2 = new Tanh()
    val Aen_3 = new Affine(100,100)
    val Ten_4 = new Tanh()

    val A_myu = new Affine(100,2)
    val A_sig = new Affine(100,2)

    //decode
    //ネットワーク構成A
    /*val Ade_1 = new Affine(20,100)
    val Tde_2 = new Tanh()
    val Ade_3 = new Affine(100,100)
    val Tde_4 = new Tanh()
    val Ade_5 = new Affine(100,784)
    val Sde_6 = new Sigmoid()
     */
    //decode
    //ネットワーク構成B
    val Ade_1 = new Affine(2,100)
    val Tde_2 = new Tanh()
    val Ade_3 = new Affine(100,784)
    val Sde_4 = new Sigmoid()
     
    //-----------学習およびテスト★-------------
    var ceArray = new Array[Double](ln)
    var errArray = new Array[Double](ln)
    var cetestArray = new Array[Double](ln)
    var errtestArray = new Array[Double](ln)
    var DklArray = new Array[Double](ln)
    var DkltestArray = new Array[Double](ln)
    var histArray2 = new Array[Double](100)
    for(i <- 0 until ln) {
      println(i+1 + "回目")
      var ce = 0d //クロスエントロピー
      var err = 0d //平均二乗誤差
      var j = 0 //画像更新の添字
      for((x,n)<-dtrain.take(dn)) { //x:入力、n:正解
        //-------学習---------
        //----------forward---------------
        val y0 = Ten_4.forward(Aen_3.forward(Ten_2.forward(Aen_1.forward(x))))
        val y_myu = A_myu.forward(y0)
        val y_sig = A_sig.forward(y0)
        val y_sigep = A_sig.forward(y0).map(_*rand.nextGaussian * 0.1)
        //A
        //val y1 = Sde_6.forward(Ade_5.forward(Tde_4.forward(Ade_3.forward(Tde_2.forward(Ade_1.forward(y_myu.zip(y_sigep).map{case(a,b)=>a+b}))))))
        //B
        val y1 = Sde_4.forward(Ade_3.forward(Tde_2.forward(Ade_1.forward(y_myu.zip(y_sigep).map{case(a,b)=>a+b}))))

        //-----------Dkl用の計算------------
        val y_myu2 = y_myu.zip(y_myu).map{case(a,b)=>a*b} //二乗
        val y_sig2 = y_sig.zip(y_sig).map{case(a,b)=>a*b}
        val myu2 = y_myu2.reduceLeft{(a,b)=>a+b} //要素値の足し合わせ
        val sig2 = y_sig2.reduceLeft{(a,b)=>a+b}

        //-------------Dkl------------------
        val Dkl = (myu2 + sig2 - 1 - math.log(sig2)/2)
        DklArray(i) = Dkl

        //---------誤差関数---------
        var d = new Array[Double](x.size)
        for(j<-0 until d.size){
          d(j) = y1(j) - x(j)
        }
              
        //-----------backward----------------
        //--------decode----------
        var d1 = new Array[Double](x.size)
        //A
        if(args(0) == "ce"){ //ce ver.
        //d1 = Ade_1.backward(Tde_2.backward(Ade_3.backward(Tde_4.backward(Ade_5.backward(d)))))
        }
        if(args(0) == "mse"){ //err ver.
        //d1 = Ade_1.backward(Tde_2.backward(Ade_3.backward(Tde_4.backward(Ade_5.backward(Sde_6.backward(d))))))
        }

        //B
        if(args(0) == "ce"){ //ce ver.
        d1 = Ade_1.backward(Tde_2.backward(Ade_3.backward(d)))
        }
        if(args(0) == "mse"){ //err ver.
        d1 = Ade_1.backward(Tde_2.backward(Ade_3.backward(Sde_4.backward(d))))
        }

        //--------encode-------------
        val d1new_myu = d1.zip(y_myu).map{case(a,b)=>a+b}

        val dDkl_sig = y_sig.map(a=>a-(1/a))
        val d1_sig = d1.map(_*rand.nextGaussian * 0.1)
        val d1new_sig = dDkl_sig.zip(d1_sig).map{case(a,b)=>a+b}

        val d_myu = A_myu.backward(d1new_myu)       
        val d_sig = A_sig.backward(d1new_sig)
        val d0 = Aen_1.backward(Ten_2.backward(Aen_3.backward(Ten_4.backward(d_myu.zip(d_sig).map{case(a,b)=>a+b}))))


        //------平均二乗誤差---------
        if(args(0) == "mse"){
          err += MSE(y1,x)
        }

        //------クロスエントロピー---------
        if(args(0) == "ce"){
          ce += CE(y1,x)
        }

        //---------update----------------
        Aen_1.update()
        Ten_2.update()
        Aen_3.update()
        Ten_4.update
        
        A_myu.update()
        A_sig.update()
        
        //A
        /*Ade_1.update()
        Tde_2.update()
        Ade_3.update()
        Tde_4.update()
        Ade_5.update()
        Sde_6.update()
         */

        //B
        Ade_1.update()
        Tde_2.update()
        Ade_3.update()
        Sde_4.update()
         

        //-----入力画像更新--------
        ds(j)=y1
        j += 1
      }

      if(args(0) == "ce"){
        ceArray(i) = ce/dn
      }
      if(args(0) == "mse"){
        errArray(i) = err/dn
      }

      //println("クロスエントロピー："+ ce /dn )
      //println("平均二乗誤差:" + err / dn )


      //------------ヒストグラム作成-----------------
      if(i == ln-1){
        for(p<-0 until 100){
          histArray2(p) = ds(p)(13+14*28)
        }
      }

      //------------画像作成--------------------
      if(i == 0 || i == ln-1 ){
        //Image.write(f"VAE-B1-" + i + ".png" , make_image(ds,10,10))
      }


      //---------テスト----------
      var cetest = 0d
      var errtest = 0d
      for((x,n) <- dtest.take(tn)) {

        //----------forward---------------
        //------encode----------
        val y0 = Ten_4.forward(Aen_3.forward(Ten_2.forward(Aen_1.forward(x))))
        val y_myu = A_myu.forward(y0)
        val y_sig = A_sig.forward(y0)
        val y_sigep = A_sig.forward(y0).map(_*rand.nextGaussian * 0.1)
        //------decode----------
        //A
        //val y1 = Sde_6.forward(Ade_5.forward(Tde_4.forward(Ade_3.forward(Tde_2.forward(Ade_1.forward(y_myu.zip(y_sigep).map{case(a,b)=>a+b}))))))
        //B
        val y1 = Sde_4.forward(Ade_3.forward(Tde_2.forward(Ade_1.forward(y_myu.zip(y_sigep).map{case(a,b)=>a+b}))))


        //-----------Dkl用の計算------------
        val y_myu2 = y_myu.zip(y_myu).map{case(a,b)=>a*b} //二乗
        val y_sig2 = y_sig.zip(y_sig).map{case(a,b)=>a*b}
        val myu2 = y_myu2.reduceLeft{(a,b)=>a+b} //要素値の足し合わせ
        val sig2 = y_sig2.reduceLeft{(a,b)=>a+b}

        //-------------Dkl------------------
        val Dkl = (myu2 + sig2 - 1 - math.log(sig2)/2)
        DkltestArray(i) = Dkl

             
        //------平均二乗誤差---------
        if(args(0) == "mse"){
          errtest += MSE(y1,x)
        }

        //------クロスエントロピー---------
        if(args(0) == "ce"){
          cetest += CE(y1,x)
        }


        //---------reset----------------
        Aen_1.reset()
        Ten_2.reset()
        Aen_3.reset()
        Ten_4.reset()

        A_myu.reset()
        A_sig.reset()

        //A
        /*Ade_1.reset()
        Tde_2.reset()
        Ade_3.reset()
        Tde_4.reset()
        Ade_5.reset()
        Sde_6.reset()
         */

        //B
       Ade_1.reset()
        Tde_2.reset()
        Ade_3.reset()
        Sde_4.reset()
     
      
      }//test

      if(args(0) == "ce"){
        cetestArray(i) = cetest/dn
      }
      if(args(0) == "mse"){
        errtestArray(i) = errtest/dn
      }

      //println("テストクロスエントロピー："+ cetest / tn )
      //println("テスト平均二乗誤差:" + errtest / tn )
     
    }//ln

    //Aen_1.save("saveB1-Aen_1")
    //Aen_3.save("saveB1-Aen_3")
    //A_myu.save("saveB1-A_myu")
    //A_sig.save("saveB1-A_sig")
    //Ade_1.save("saveB1-Ade_1")
    //Ade_3.save("saveB1-Ade_3")
    //Ade_5.save("saveB0-Ade_5")


    println("<<<<<-B0->>>>>")

/*
    if(args(0) == "mse"){
      println("-----MSE-----")
      for(i<-0 until ln){
        print(errArray(i) + ",")
      }
      println(" ")
      println("-----testMSE-----")
      for(i<-0 until ln){
        print(errtestArray(i) + ",")
      }
    }

    if(args(0) == "ce"){
      println("-----CE-----")
      for(i<-0 until ln){
        print(ceArray(i) + ",")
      }
      println(" ")
      println("-----testCE-----")
      for(i<-0 until ln){
        print(cetestArray(i) + ",")
      }
    }

    println(" ")
    println("-----Dkl-----")
    for(i<-0 until ln){
      print(DklArray(i) + ",")
    }

    println(" ")
    println("-----testDkl-----")
    for(i<-0 until ln){
      print(DkltestArray(i) + ",")
    }*/

    println(" ")
    println("----hist_before------")
    for(i<-0 until 100){
      print(histArray1(i) + ",")
    }

    println(" ")
    println("----hist_after------")
    for(i<-0 until 100){
      print(histArray2(i) + ",")
    }



  }//main
}//VAE
