import cv2 as cv
import numpy as np
from skimage import draw
from scipy.interpolate import UnivariateSpline as spline
import matplotlib.pyplot as plt
import time 

#  Helper Classes ====================================================================================================================================


class Ellipse():
    x: float = 0
    y: float = 0
    La: float = 0
    Lb: float = 0
    alpha: float = 0


class EllipseParameter():
    suchweite: int = 25
    strahlen: int = 100
    interpolation: float = 0.25


# ====================================================================================================================================================


def findHighPrecisionTargets(undist_img, gray):
    # determine_approximation_values:
    #  finde alle Kandidaten für eine genaue Untersuchung mit folgenden Ansprüchen:
    #   - schnell
    #   - eher mehr als zu wenige Kandidaten
    #       -> findORB(draw_img, gray)   -> for edges                    -> not useful
    #       -> HoughCircles                -> for circles                  -> not useful
    #       -> findSift(draw_img, gray)  -> for inner contour-points     -> particular useful and slow
    # -> investigate trafic sign detection
    #     -> https://www.kaggle.com/code/valentynsichkar/traffic-signs-detection-by-yolo-v3-opencv-keras/notebook
    #     -> https://www.youtube.com/watch?v=yGMZOD44GrI
    #     -> https://www.youtube.com/watch?v=n9_XyCGr-MI
    #     -> https://github.com/aladdinpersson/Machine-Learning-Collection


    # 1. Liste of candidates
    approximation_values = determine_approximation_values(undist_img, gray, draw=True)

    # 2. Ellipse fitting in two runs
    if len(approximation_values) > 1:
        for point in approximation_values:
            # ---------------------------------------------------------------------- map to ellipse parameter
            ellipse = Ellipse()
            ellipse.x = point[0]
            ellipse.y = point[1]
            ellipse.La = point[2] 
            ellipse.Lb = point[2]
            ellipse.alpha = point[3]

            # --------------------------------------------------------------- 2 runs for the ellipse measurement
            approximated_ellipse = get_ellipse_approximation(gray, ellipse, False)
            # final_ellipse = get_ellipse_approximation(gray, approximated_ellipse, True)



    # hoch genaue Mittelpunktsbestimmung
    # -> nehme Kandidat
    # -> Bilde Strahlen ausgehend vom Mittelpunkt radial (50 Strahlen) nach außen
    # -> bestimme auf den Strahlen den ersten Hell/Dunkel-Übergang
    # -> fitte in alle Kantenpunkte eine ausgleichende Ellipse
    # -> Ausreißerkontrolle
    # -> berechne Ergebnis
    return ellipse
    pass


def get_ellipse_approximation(gray_img, ellipse, draw):
    parameter = EllipseParameter()  # standard parameter
    parameter.suchweite = ellipse.La * 1.25   # maximum 1.25

    search_rays = get_serch_coordinates(ellipse, parameter)

    # pixel_values = get_pixel_values(gray_img, search_rays)
    #
    # corner_points= get_corner_points(pixel_values)
    #
    # fit_ellipse = get_ellipse_fit(gray_img)
    
    corners = search_ray(undist_img, gray_img, ellipse.x, ellipse.y, suchweite, strahlen)

    res_ellipse = Ellipse()
    res_ellipse = ellipse

    return res_ellipse

def search_ray(undist_img, gray, x, y, suchweite, strahlen):
    list_corner = []
    # Lenght of rays
    length = suchweite * 1.25
    # Test with chosen middle point
    for i in range(strahlen):
        # computing line with different angles 
        angle = i * (360/strahlen)
        # rad
        angle*= np.pi/180
        # Start point
        x = int(np.round(x))
        y = int(np.round(y))
        # End point
        x_2 =  int(np.round(x + length * np.cos(angle)))
        y_2 =  int(np.round(y + length * np.sin(angle)))
        
        # Draw a line for a visualisation
        cv.line(undist_img,(x,y),(x_2,y_2),(255,255,255),1)

        # Getting pixel values from ray
        gray_pixel_value,ray_coordinates = get_pixel_values(gray, x, y, x_2, y_2)
        # Finding corners
        corners_coordinates = get_corner_points(gray_pixel_value,ray_coordinates)

    list_corner.append(corners_coordinates)

    return list_corner

def get_corner_points(pixel_values,ray_coordinates):
    """
    calculates the first and the second derivative of the pixel values and estimates the edge points
    """
    x = np.arange(len(pixel_values))
    y = pixel_values

    # Fit a function to values, k Degree of the smoothing spline. Must be 1 <= k <= 5. k = 3 is a cubic spline. S smoothing factor used to choose the number of knots
    y_spl = spline(x,y,k=3,s=300)
    
    y_spl_1d = y_spl.derivative(n=1)
    y_spl_2d = y_spl.derivative(n=2)
    d1_value = y_spl_1d(x)
    d2_value = y_spl_2d(x)
 
    
    # Positive slope = black to white, negative slope = white to black
    local_max_min_idx = []
    slope = 10
    if np.any(d1_value > slope):
        steepest_slope = list(d1_value).index(max(d1_value))
        local_max_min_idx.append(steepest_slope)
    if np.any(d1_value < -slope):
        steepest_slope = list(d1_value).index(min(d1_value))
        local_max_min_idx.append(steepest_slope)

    return ray_coordinates[local_max_min_idx]
        
    #for i in range(len(d1_value)):
    #    if d1_value[i] > 10:
    #        positive_slope_idx= list(d1_value).index(d1_value[i])
    #        local_max_min_idx.append(positive_slope_idx)
    #    if d1_value[i] < -10: 
    #        negative_slope_idx = list(d1_value).index(d1_value[i])
    #        local_max_min_idx.append(negative_slope_idx)
    
def get_pixel_values(gray, x, y, x_2, y_2):
    """
    grab the interpolated pixel values
    """
    pass


def get_serch_coordinates(ellipse: Ellipse, search_para: EllipseParameter):
    """
    calculates coordinates of the search rays
    """
    x = ellipse.x
    y = ellipse.y
    La = ellipse.La
    Lb = ellipse.Lb
    alpha = ellipse.alpha

    Suchweite = search_para.suchweite
    Schrittweite = search_para.interpolation
    Strahlen = search_para.strahlen

    ## test ##

    StartPunkt = np.trunc(Lb * (1 - Suchweite) / Schrittweite)
    EndPunkt = np.trunc(La * (1 + Suchweite) / Schrittweite) + 1

    dw = 2 * np.pi / Strahlen                                     # Winkelabstand der Strahlen im Kreis [rad]
    # ----------------------------------------------------------------- Brennpunkt ausrechnen = > e ^ 2 = a ^ 2 - b ^ 2

    e = np.sqrt(abs(np.square(La) - np.square(Lb)))

    F1 = rotate_2d([e, 0], [0, 0], np.rad2deg(alpha))
    F2 = rotate_2d([-e, 0], [0, 0], np.rad2deg(alpha))

    
    for n in range(Strahlen):
        x = La * np.cos(dw * n)           # Richtung Bildstrahl, lokal in Ellipse
        y = Lb * np.sin(dw * n)
        Re = rotate_2d([x, y], [0, 0], np.rad2deg(alpha))

        # -------------------------------------- ggf. https://pypi.org/project/transformations/ oder https://github.com/dfki-ric/pytransform3d

        R1 = GetNormVec(Re - F1)
        R2 = GetNormVec(Re - F2)
        Rges = GetNormVec(R1 + R2)
        
        # -------------------------------------------------------------------------------------------------------
        #x_rges = x  + Re[0] + 1 * Schrittweite * Rges[0]
        #y_rges = y  + Re[1] + 1 * Schrittweite * Rges[1]
        #x_list = [0,x,x_rges,Re[0]]
        #y_list = [0,y,y_rges,Re[1]]
        #plt.plot(x_list,y_list,'bo', linestyle="--")
        #plt.show()

        # # -------------------------------------------------------------------------------------------------------
        lmin = round(-Suchweite * GetBetrag(Re) / Schrittweite)
        lmax = lmin + abs(StartPunkt - EndPunkt)
          
        k = 0
        BP = []
        xoo = x  + Re[0] + 1 * Schrittweite * Rges[0]
    #     for t:= lmin  to lmax  do begin  // Pixel auslesen
    #       BP[k].x:=  x  + Re.x + t * Schrittweite * Rges.x;                   // X-Koordinate
    #       BP[k].y:=  y  + Re.y + t * Schrittweite * Rges.y;                   // Y-Koordinate
    #       BP[k].z:=  Get_Color_Bilinear( Bild,  BP[k].x,  BP[k].y, color );   // Grauwert
    #       inc(k);
    #     end;

    #     setlength(BP, k);
    #     erg.Rand[n] := BP;
    #   end;
    # end;



def rotate_2d(p, origin=(0, 0), degrees=0):
    """
        -> https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    """
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def GetNormVec(x, y):
    l = np.sqrt(np.square(x) + np.square(y) )
    if l > 0:
        res_x = x / l
        res_y = y / l
    else:
        res_x = 0
        res_y = 0

    return [res_x, res_y]


def GetBetrag(x, y):
    return np.sqrt(np.square(x) + np.square(y))



def get_color_bilinear(gray_img, x, y):
    """
    bilinear inperolation in an gray image(gray_img) at the position (x, y)
    """

    ax = int(np.floor(x))
    ay = int(np.floor(y))

    u = x - ax
    v = y - ay

    a00 = gray_img[ay, ax]                 # Helligkeitswert links oben
    a01 = gray_img[ay + 1, ax]             # Helligkeitswert  rechts oben
    a10 = gray_img[ay, ax + 1]             # Helligkeitswert links  unten
    a11 = gray_img[ay + 1, ax + 1]         # Helligkeitswert rechts unten

    d =     a00 * (1.0 - u) * (1.0 - v)
    d = d + a01 *       (u) * (1.0 - v)
    d = d + a10 * (1.0 - u) *       (v)
    d = d + a11 *       (u) *       (v)

    return d                                # bilinear interpolierter Helligkeitswert

def get_neighboor_pixel_value(gray, cx, cy):
    # https://stackoverflow.com/questions/53334694/get-perimeter-of-pixels-around-centre-pixel
    width = gray.shape[1]
    height = gray.shape[0]
    radius = 2
    
    x = np.arange(cx - radius, cx + radius + 1)
    y_off = np.sqrt(radius**2 - (x - cx) **2)
    y1 = np.int32(np.round(cy + y_off))
    y2 = np.int32(np.round(cy - y_off))

    # if x,y is within image shape 
    if all(x_v < width for x_v in x) and  all(y_v < height for y_v in y1)  and  all(y_v < height for y_v in y2):    
        return gray[y1,x], gray[y2,x]  
    # else return empty array
    return np.zeros((len(x))) + 1, np.zeros((len(x))) + 1  

# ==================================================================================================================================================
def determine_approximation_values(img, gray, draw=True):
    """
    gray_img    ->  drawing image
    gray   ->  measuring image
    return ->  list of [position.x, position.y, feature.size, feature.angle]
    """

    # ----------------------------------------------------------------------------
    # app_val = list of [feature.x, feature.y, feature.size, feature.angle]
    # app_val = findSift(img, gray, draw)
    # app_val = findORB(img, gray, draw)
    app_val = HoughCircles(img, gray, draw)
    # app_val = blob(img, gray, draw)
    # app_val = haarcascade(img, gray, draw) # TODO: not working!
    # app_val = SimpleBlobDetector(img,gray, draw=True) # not working well!


    return app_val

# -------------------------------------------------------------------------------- helper operators for "Näherungswerte"

def haarcascade(img, gray, draw=True):
    # Importing Haar cascade classifier xml data -> to create http://note.sonots.com/SciSoftware/haartraining.html
    xml_data = cv.CascadeClassifier('XML-data.xml')
    # Detecting object in the image with Haar cascade classifier
    detecting = xml_data.detectMultiScale(gray, minSize=(30, 30))
    # Amount of object detected
    amountDetecting = len(detecting)
    # Using if condition to highlight the object detected
    if amountDetecting != 0:
        for (a, b, width, height) in detecting:
            cv.rectangle(img, (a, b),  # Highlighting detected object with rectangle
                          (a + height, b + width),
                          (0, 275, 0), 9)
            # Plotting image with subplot() from plt

    app_val = []
    return app_val

def HoughCircles(img, gray, draw=True):
    # https://stackoverflow.com/questions/59363165/detect-multiple-circles-in-an-image

    width = img.shape[1]
    height = img.shape[0]
    param1 = 180
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT,
                              minDist=1,
                              dp=1,
                              param1=180,
                              param2=5,
                              minRadius=2,
                              maxRadius=15)
    
    # For all type of lightning
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 27, 3)
    
    app_val = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
                #cv.circle(img, (x, y), 1, (0, 255, 0), -1)
                #app_val.append([x, y, r, 0])

                # Filtering center points
                neighboor_pixel, neighboor_pixel_2 = get_neighboor_pixel_value(thresh,x,y)
                # If center point has neighboors with higher threshhold than 0
                if all(value == 0 for value in neighboor_pixel) and  all(value == 0 for value in neighboor_pixel_2):
                    cv.circle(img, (x, y), 1, (0, 255, 0), -1)
                    app_val.append([x, y, r, 0]) 
                                 
    print(len(app_val))
    return app_val


def blob(img, gray, draw=True):

    # Threshhold filtering
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 27, 3)
    
    #plt.imshow(cv.cvtColor(thresh, cv.COLOR_BGR2RGB))
    #plt.show()

    cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Init variables
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    count = 0
    app_val = []
    circularity_ellipse = False

    # Filtering contour
    for c in cnts:
        area = cv.contourArea(c)
        if area > 0:
            # Perimeter of contour
            perimeter = cv.arcLength(c, True)
            # Bounding Rectangular
            x_rect, y_rect, w, h = cv.boundingRect(c)
            # Bounding Circle
            ((x, y), r) = cv.minEnclosingCircle(c)

            # --------------------------------------------------------------------------------
            # Filtering by convexity
            hull = cv.convexHull(c)
            hull_area = cv.contourArea(hull)
            convexity = area/hull_area
            convexity_range = 0.5 < convexity < 1

            # Filtering by ratio of width and height for ellipse
            ratio = w/h
            ellipse_ratio = (ratio > 0.5) and (ratio < 1.5)
 
            # Filtering by Area 
            area_filter = 5 < hull_area < 1000
            # Filtering by Circularity
            circularity = 4*np.pi*(hull_area/(perimeter*perimeter))
            circularity_ellipse = 0.75 < circularity < 1

            if area_filter and circularity_ellipse:
                app_val.append([x, y, max(w, h), 0])  # definition x, y, diameter, alpha
                M = cv.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv.circle(img, (int(cX), int(cY)), int(r), (36, 255, 12), 1)
                count += 1
            
    print('Number of candidates: ', count)
    return app_val

def SimpleBlobDetector(img,gray, draw=True):

    # https://docs.opencv.org/3.4/d0/d7a/classcv_1_1SimpleBlobDetector.html, https://learnopencv.com/blob-detection-using-opencv-python-c/
    params = cv.SimpleBlobDetector_Params() 
    params.minThreshold = 0
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01


    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs.
    app_val = detector.detect(img)
    print(len(app_val))

    if draw:
        cv.drawKeypoints(img, app_val, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return 
# -------------------------------------------------------------------------------- helper operators for "Näherungswerte"
def findORB(img, gray, draw=True):
    # https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
    orb = cv.ORB_create(500000)
    kp, descr = orb.detectAndCompute(gray, None)
    if draw:
        cv.drawKeypoints(img, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    shape = img.shape
    app_val = []
    for point in kp:
        if (point.pt[0] > 1) and (point.pt[0] < shape[0] - 1) and (point.pt[1] > 1) and (point.pt[1] < shape[1] - 1):
            app_val.append([point.pt[0], point.pt[1], point.size, point.angle])

    return app_val


def findSift(img, gray, draw=True):
    # https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

    min_size = 5
    max_size = 50
    sift = cv.SIFT_create(50000)
    kp = sift.detect(gray, None)

    shape = img.shape
    filtered_kp = []
    for point in kp:
        if (min_size < point.size < max_size) and (point.pt[0] > 1) and (point.pt[0] < shape[1]-1) and (point.pt[1] > 1) and (point.pt[1] < shape[0] - 1):
            filtered_kp.append(point)


    app_val = []
    for point in filtered_kp:
        app_val.append([point.pt[0], point.pt[1], point.size, point.angle])

    if draw:
        cv.drawKeypoints(img, filtered_kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return app_val




def Mittelpunktsbestimmung():
    return


if __name__ == '__main__':
    print("unit for measure_precision_target detection \n \t -> run facial_landmarks_pose_detection.py  \n \t    with option measure_measure_precision_targets=True")
    
    path = r'D:\Nguyen\Test_Images\230a3894.jpg'
    img = cv.imread(path)

    width,height = 2560, 1440 # Auflösungen:  320, 180, 640, 360, 1280, 720, 1920, 1080, 2560, 1440, 3840,2160
    img = cv.resize(img,(width,height),cv.INTER_AREA)
    print(img.shape[:2])

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #findSift(img,gray)
    #findHighPrecisionTargets(img,gray)
    start = time.time()
    determine_approximation_values(img,gray,draw=True)
    print('Time: ' ,time.time()- start)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

    #cv.imshow("Test",img)
    cv.waitKey(0)
