import math

def placement(xval, yval, prev, ax):
    if math.fabs(xval - ax.get_xlim()[1]) > 1:
        x_neg = False
        x_step = 1.5
    else:
        x_neg = True
        x_step = -1.5
    if math.fabs(yval - ax.get_ylim()[1]) > 1:
        y_step = 0.5
    else:
        y_step = -0.5
    c_r = 0.45 #Trial and error value
    iter = 0
    coll = False #Is there a collision?

    while True:

        iter += 1

        for p in prev:
            #Simple radius-based collision detection
            distx = (xval + x_step + c_r)-(p[0]+p[2]+c_r)
            disty = (yval + y_step + c_r)-(p[1]+p[3]+c_r)
            dist = math.sqrt((distx * distx) + (disty * disty))
            if dist <= (2*c_r):
                coll = True
                break #No point in checking the rest

        if coll == False:
            break
        
        coll = False
        if not x_neg:
            x_step += 1
        else:
            x_step -= 1

    if iter > 1:
        print 'Text Collision: px:{} py:{} x:{} y:{}'.format(p[0],p[1],
              xval,yval)
        print 'Number of iterations to fix : {}'.format(iter-1)

    return x_step, y_step

