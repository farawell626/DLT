import numpy as np
import matplotlib.pyplot as plt
import ast
from mpl_toolkits.mplot3d import Axes3D
from Parabola import ParabolaSequence

def ThreeDApproximate(P: np.array,path):
    '''
    給2D點與對應時間算出6未知X0 Y0 Z0 Vx Vy Vz
    Parameters
    --------------
    P : list
        相機參數
    point : numpy array
        2D點
    t : integer
        時間(幀)  

    Returns
    --------------
    Tr: the transformation matrix (translation plus scaling)
    x: the transformed data
    '''
    g = -9.8
    #點資料處理
    pointlist,pointlist_ynorm = ReadTxt(path)

    x=[]
    y=[]
    for point in pointlist:
        x.append(point[0])
        y.append(point[1])
    
    xDraw=[]
    yDraw=[]
    for p in pointlist_ynorm:
        xDraw.append(p[0])
        yDraw.append(p[1])
    
    #connectPoint = FindConnectedPoint(pointlist_ynorm)
    #connectPoint = [ 23, 51, 94, 130, 153]
    connectPoint =  [26, 53, 91, 121, 133] #02
    print (connectPoint)
    #畫圖
    '''
    Drawline(xDraw,yDraw,pointlist_ynorm)
    Drawline(x,y,pointlist)
    '''

    # 每段點數量
    SegAmount = []
    for i in range(0,len(connectPoint)-1):
        SegAmount.append(connectPoint[i+1] - connectPoint[i])
    print("SegAmount: " + str(SegAmount))
    firstSeg = connectPoint[1] - connectPoint[0]  
    secondSeg = connectPoint[2] - connectPoint[1]  
    thirdSeg = connectPoint[3] - connectPoint[2] 

    # All segment ans (ParabolaSequence list)
    ansList = []

    for seg in range(0,len(SegAmount)-1):
        # left * ans = right  ->  ans = inv(left) * right
        left = []
        right = []
        for i in range(0,SegAmount[seg]):
            tn = (i*0.033)
            xn = pointlist[connectPoint[seg]+1 + i][0]
            yn = pointlist[connectPoint[seg]+1 + i][1]

            #左邊的Set 2*n
            lxTerm = [
                [P[0][0]-xn*P[2][0]],
                [tn*(P[0][0]-xn*P[2][0])],
                [P[0][1]-xn*P[2][1]],
                [tn*(P[0][1]-xn*P[2][1])],
                [P[0][2]-xn*P[2][2]],
                [tn*(P[0][2]-xn*P[2][2])]
            ]

            lyTerm = [
                [P[1][0]-yn*P[2][0]],
                [tn*(P[1][0]-yn*P[2][0])],
                [P[1][1]-yn*P[2][1]],
                [tn*(P[1][1]-yn*P[2][1])],
                [P[1][2]-yn*P[2][2]],
                [tn*(P[1][2]-yn*P[2][2])]
            ]
            left.append(lxTerm)
            left.append(lyTerm)

            #右邊的Set 2*n
            rxTerm = [xn*((P[2][2]*g*pow(tn,2)/2.0)+1) - ((P[0][2]*g*pow(tn,2))/2.0+P[0][3])]
            ryTerm = [yn*((P[2][2]*g*pow(tn,2)/2.0)+1) - ((P[1][2]*g*pow(tn,2))/2.0+P[1][3])]
            right.append(rxTerm)
            right.append(ryTerm)

        rightNPA = np.array(right)
        leftNPA = np.array(left)
        leftNPA = leftNPA.reshape(SegAmount[seg]*2,6)

        ans = np.dot(np.linalg.pinv(leftNPA),rightNPA)
        print("Segment {} ({} to {}): ".format(seg,connectPoint[seg]+1,connectPoint[seg+1]))
        print(ans)
        ansList.append(ParabolaSequence(ans,SegAmount[seg]))

    #DrawPara(ansList)
    #DrawCourt2()
    return 0

def DrawXTPYTP(P: np.array,path):
    g = -9.8
    #點資料處理
    pointlist,pointlist_ynorm = ReadTxt(path)

    x=[]
    y=[]
    for point in pointlist:
        x.append(point[0])
        y.append(point[1])
    
    xDraw=[]
    yDraw=[]
    for p in pointlist_ynorm:
        xDraw.append(p[0])
        yDraw.append(p[1])

    #畫圖
    
    Drawline(xDraw,yDraw,pointlist_ynorm)
    Drawline(x,y,pointlist)   

def Drawline(x,y,pointlist):

        
    plt.plot(x)
    plt.plot(y,'r.')

    for turningPoint in FindConnectedPoint(pointlist):
        plt.plot(turningPoint,pointlist[turningPoint][1],'b.')

    plt.title("XTP & YTP") # title

    plt.show()
    
    return 0

def ReadTxt(path):
    
    pointlist = []
    pointlist_ynorm = []
    with open(path) as f:
        line = f.read()
        s = line.split('\n')
        
        wh = s.pop(0)
        frame = s.pop(0)
        w = wh.split(',')[0][1:]
        h = wh.split(',')[1][1:-1]
        for id in s:
        
            id=id[1:-1]
            id = id.replace(" ","")
            a = id.split(',')
            
            p = [int(a[0]),int(a[1])]
            p_ynorm = [int(a[0]),abs(int(a[1])-int(h))]
            #print(p)abs(int(a[1])-1080)
            pointlist.append(np.array(p))
            pointlist_ynorm.append(np.array(p_ynorm))
    #validation
    for i in range(len(pointlist)):
        if (pointlist[i][0] == 0 or pointlist[i][1] == 0) and i != 0:
            count = 1
            for k in range(i+1,len(pointlist)):
                count+=1
                if(pointlist[k][0] == 0 or pointlist[k][1] == 0):
                    continue
                else:
                    
                    pointlist[i][0] = (pointlist[k][0] - pointlist[i-1][0]) / count + pointlist[i-1][0]
                    pointlist[i][1] = (pointlist[k][1] - pointlist[i-1][1]) / count + pointlist[i-1][1]
                    break

    for i in range(len(pointlist_ynorm)):
        
        if pointlist_ynorm[i][0] == 0 or pointlist_ynorm[i][1] == 0  :
            count = 1
            
            for k in range(i+1,len(pointlist_ynorm)):
                count+=1
                
                if(pointlist_ynorm[k][0] == 0 or pointlist_ynorm[k][1] == 0):
                    continue
                else:
                    
                    pointlist_ynorm[i][0] = (pointlist_ynorm[k][0] - pointlist_ynorm[i-1][0]) / count + pointlist_ynorm[i-1][0]
                    pointlist_ynorm[i][1] = (pointlist_ynorm[k][1] - pointlist_ynorm[i-1][1]) / count + pointlist_ynorm[i-1][1]
                    break
  

    return pointlist,pointlist_ynorm

def FindConnectedPoint(points:list):
    index = []
    for i in range(2,len(points)-2):
        if(i<=0 or i>=len(points)-1):
            continue
        # 算斜率
        m0 = points[i][1] - points[i-1][1]
        m1 = points[i+2][1] - points[i][1]
        if(m0<0 and m0*m1 <=0):
            index.append(i)

    return index

def Parabola(pWithV: ParabolaSequence, mode:int = 0):
    '''
        Z
        |
        |_____X
       /
      /
     Y 
    Z軸是向上
    pWithV: X Vx Y Vy Z Vz

    Xt = X0 + VX * t
    Yt = Y0 + VY * t
    Zt = Z0 + VZ * t + g*t*t / 2
    '''
    g = -9.8
    Xo = pWithV.X0()
    Yo = pWithV.Y0()
    Zo = pWithV.Z0()
    Vx = pWithV.Vx()
    Vy = pWithV.Vy()
    Vz = pWithV.Vz()

    paraPoints = []
    for t in range(0,pWithV.pAmount()):
        if t % 5 == 0:
            tN = t*0.033
            if mode == 0:
                paraPoints.append(
                    [
                        round(Xo + Vx*tN, 3) ,
                        round(Yo + Vy*tN, 3),
                        round(Zo + Vz*tN + g*tN*tN/2.0, 3)
                    ])
            elif mode == 1:
                paraPoints.append(
                    [
                        round(Yo + Vy*tN, 3),
                        round(Zo + Vz*tN + g*tN*tN/2.0, 3),
                        round(Xo + Vx*tN, 3)
                    ])


    return paraPoints

def DrawPara(ans:list):  
    #print(ans[0].data)   
    for i in range(0,len(ans)):
        points = Parabola(ans[i])
        print (points)

    ax = plt.figure().add_subplot(projection='3d')    
    '''
    for point in points:
        ax.scatter(point[1], point[0], point[2], c='r',marker='o', label='My Points 2')
        ax.scatter(points[0][1], points[0][0], points[0][2], c='b',marker='o', label='My Points 2')
    ax.set_xlabel('Y Label')
    ax.set_ylabel('X Label')
    ax.set_zlabel('Z Label')
    plt.show()
    '''
    return
    
def SequenceFitting(P: np.array,path):
    '''
    由於每個3D子軌跡都是獨立近似的
    所以根據前一子軌跡計算的一對相鄰子軌跡之間的過渡點的3D座標並不總是與根據後一子軌跡計算的一致
    為了克服這個問題
    我們透過同時考慮兩個相鄰的子軌跡來增強演算法

    Xt = X1 - V_X0 * t
    Yt = Y1 - V_Y0 * t
    Zt = Z1 - V_Z0 * t - g*t*t / 2

    Xt = X1 + V_X1 * t
    Yt = Y1 + V_Y1 * t
    Zt = Z1 + V_Z1 * t + g*t*t / 2

    Parameters
    --------------
    P : list
        相機參數
    point : numpy array
        2D點
    t : integer
        時間(幀)  

    Returns
    --------------
    Tr: the transformation matrix (translation plus scaling)
    x: the transformed data
    '''
    g = -9.8

    #點資料處理
    pointlist,pointlist_ynorm = ReadTxt(path)

    x=[]
    y=[]

    for point in pointlist:
        x.append(point[0])
        y.append(point[1])
    
    xDraw=[]
    yDraw=[]
    for p in pointlist_ynorm:
        xDraw.append(p[0])
        yDraw.append(p[1])
    
    #connectPoint = FindConnectedPoint(pointlist_ynorm)

    #connectPoint = [0, 23, 51, 94, 130, 153]
    connectPoint =  [0, 26, 53, 91, 121, 133]#02
    print (connectPoint)
    #畫圖
    '''
    Drawline(xDraw,yDraw,pointlist_ynorm)
    Drawline(x,y,pointlist)
    '''

    # 每段點數量
    # SegAmount: [54, 66, 69, 43]
    SegAmount = []
    for i in range(0,len(connectPoint)-2):
        SegAmount.append(connectPoint[i+2] - connectPoint[i]+1)
    print("SegAmount: " + str(SegAmount))

    '''
    發球P1，以0幀~53幀 0~26 27~53 K=26 N=53
    共54幀計算

    傳球P2，以26幀~91幀 26~53 54~91
    共66幀計算

    舉球P3，以53幀~121幀 53~91 92~121
    共69幀計算

    殺球P4，以91幀~133幀 91~121 122~133
    共43幀計算
    '''
    # All segment ans (ParabolaSequence list)
    ansList = []

    for seg in range(0,len(SegAmount)):
        # left * ans = right  ->  ans = inv(left) * right
        left = []
        right = []
        K = connectPoint[seg + 1]
        for i in range(0,SegAmount[seg]):
            tn = (abs(i - (K - connectPoint[seg]))* 0.033)
            #print(str(K) + " " + str(abs(i - (K - connectPoint[seg]))) + " " + str(i))
            #print(round(SegAmount[seg]/2))
            xn = pointlist[connectPoint[seg] + i][0]
            yn = pointlist[connectPoint[seg] + i][1]
            #print(str(connectPoint[seg] + i) + " " + str(K))
            # 0 ~ 2K term
            if (connectPoint[seg] + i) < K:
                
                #左邊的Set 2*k
                lxTerm = [
                    [P[0][0]-xn*P[2][0]],
                    [tn*(P[0][0]-xn*P[2][0])],
                    [0],

                    [P[0][1]-xn*P[2][1]],
                    [tn*(P[0][1]-xn*P[2][1])],
                    [0],

                    [P[0][2]-xn*P[2][2]],
                    [tn*(P[0][2]-xn*P[2][2])],
                    [0]
                ]

                lyTerm = [
                    [P[1][0]-yn*P[2][0]],
                    [tn*(P[1][0]-yn*P[2][0])],
                    [0],

                    [P[1][1]-yn*P[2][1]],
                    [tn*(P[1][1]-yn*P[2][1])],
                    [0],

                    [P[1][2]-yn*P[2][2]],
                    [tn*(P[1][2]-yn*P[2][2])],
                    [0]
                ]
                left.append(lxTerm)
                left.append(lyTerm)

                #右邊的Set 2*k
                rxTerm = [xn*((P[2][2]*g*pow(tn,2)/2.0)+1) - ((P[0][2]*g*pow(tn,2))/2.0+P[0][3])]
                ryTerm = [yn*((P[2][2]*g*pow(tn,2)/2.0)+1) - ((P[1][2]*g*pow(tn,2))/2.0+P[1][3])]
                right.append(rxTerm)
                right.append(ryTerm)           
            elif (connectPoint[seg] + i) >= K:
                #左邊的Set 2*(n-k)
                lxTerm = [
                    [P[0][0]-xn*P[2][0]],
                    [0],
                    [tn*(P[0][0]-xn*P[2][0])],

                    [P[0][1]-xn*P[2][1]],
                    [0],
                    [tn*(P[0][1]-xn*P[2][1])],

                    [P[0][2]-xn*P[2][2]],
                    [0],
                    [tn*(P[0][2]-xn*P[2][2])] 
                ]

                lyTerm = [
                    [P[1][0]-yn*P[2][0]],
                    [0],
                    [tn*(P[1][0]-yn*P[2][0])],
                    
                    [P[1][1]-yn*P[2][1]],
                    [0],
                    [tn*(P[1][1]-yn*P[2][1])],
                    
                    [P[1][2]-yn*P[2][2]],
                    [0],
                    [tn*(P[1][2]-yn*P[2][2])]   
                ]
                left.append(lxTerm)
                left.append(lyTerm)

                #右邊的Set 2*(n-k)
                rxTerm = [xn*((P[2][2]*g*pow(tn,2)/2.0)+1) - ((P[0][2]*g*pow(tn,2))/2.0+P[0][3])]
                ryTerm = [yn*((P[2][2]*g*pow(tn,2)/2.0)+1) - ((P[1][2]*g*pow(tn,2))/2.0+P[1][3])]
                right.append(rxTerm)
                right.append(ryTerm)
            
        
        rightNPA = np.array(right)
        leftNPA = np.array(left)
        print(len(left))
        print(SegAmount[seg] * 2)
        
        #leftNPA = leftNPA.reshape((K - connectPoint[seg]) * 2,6)
        leftNPA = leftNPA.reshape(SegAmount[seg] * 2,9)
        

        ans = np.dot(np.linalg.pinv(leftNPA),rightNPA)
        print("Segment {} ({} to {}): ".format(seg,connectPoint[seg],connectPoint[seg+2]))
        print(ans)
        s0 = [ans[0],ans[1],ans[3],ans[4],ans[6],ans[7]]
        s1 = [ans[0],ans[2],ans[3],ans[5],ans[6],ans[8]]
        #print(s0[0].item())
        ansList.append(ParabolaSequence(s0,K - connectPoint[seg]))
        ansList.append(ParabolaSequence(s1, connectPoint[seg+2] - K + 1))
    print(len(ansList))

    #DrawPara(ansList)
    #DrawCourt2()
    return 0    

def DrawCourt():
    # create x,y
    y = [-900   , 900   , 900   , -900, -900]
    x = [0      , 0     , 900   , 900, 0]
    z = 0  # 地板的z坐标

    # plot the surface
    #plt3d  = plt.subplot(projection='3d')
    ax = plt.axes(projection='3d')
    ax.plot(y, x, z) 
    ax.set_xlabel('Y Label')
    ax.set_ylabel('X Label')
    ax.set_zlabel('Z Label')
    #plt3d.plot_surface(xx, yy, z)

    plt.show()

def DrawCourt2():
    # 地板邊緣的x和y坐標
    y = np.array([-900, 900, 900, -900, -900])
    x = np.array([0, 0, 900, 900, 0])
    
    # 創建表示地板的z坐標，這裡z是一個常數
    z = 0
    
    # 創建一個3D軸
    ax = plt.axes(projection='3d')
    
    # 使用plot3D來繪製邊界
    ax.plot3D(y, x, z*np.ones(y.shape), 'gray')
    
    # 使用plot_surface來繪製地板，首先需要創建一個網格
    Y, X = np.meshgrid(y, x)  # 創建網格
    Z = z * np.ones(Y.shape)  # 所有點的Z坐標都是0，因為地板是平的
    
    # 繪製地板
    ax.plot_surface(Y, X, Z, alpha=0.5)  # alpha用於設置透明度
    
    # 設置軸標籤
    ax.set_xlabel('Y Label')
    ax.set_ylabel('X Label')
    ax.set_zlabel('Z Label')

    # 顯示圖形
    plt.show()

#def HandleOulier(pointlist:list):



