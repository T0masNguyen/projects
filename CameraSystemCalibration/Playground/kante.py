#https://www.scipy-lectures.org/intro/summary-exercises/optimize-fit.html

from scipy.optimize import leastsq
import numpy as np
from matplotlib import pyplot as plt

def modelGauss(t, coeffs):
	return coeffs[0] + coeffs[1] * np.exp( - ((t - coeffs[2])/coeffs[3])**2 )
#coeffs[0] is B (noise)
#coeffs[1] is A (amplitude)
#coeffs[2] is \mu (center)
#coeffs[3] is \sigma (width)
# y(t) = B + A*exp(-( (t - mu)/sigma )^2)

def residuals(coeffs, y, t):
	return y - modelGauss(t, coeffs)

def ableitungsfilter(profil):
	if(len(profil) < 8):
		'zu kurz'
		return np.arange(1)
	
	filterMask7 = np.array([-0.03125, -0.125, -0.15625, 0.0, 0.15625, 0.125, 0.03125])
	
	filteredValues = np.arange(len(profil) - 6)
	
	for i in range(len(filteredValues)):
		filteredValues[i] = 0
		for j in range(len(filterMask7)):
			filteredValues[i] = filteredValues[i] + filterMask7[j]*profil[i+j]
	
	return filteredValues



def initialValues(profilGauss):
	x0 = np.array([min(profilGauss),\
	 max(profilGauss) - min(profilGauss), np.argmax(profilGauss),\
	  0.25*len(profilGauss)], dtype=float)
	return x0
#x0[0] is B (noise)
#x0[1] is A (amplitude)
#x0[2] is \mu (center)
#x0[3] is \sigma (width)
# y(t) = B + A*exp(-( (t - mu)/sigma )^2)

profil1 = np.array([40, 40, 40, 40, 40, 40, 65, 65, 65, 65, 65, 65, 65], dtype=float)

profil2 = np.array([40, 40, 40, 40, 40, 40, 47.5, 65, 65, 65, 65, 65, 65], dtype=float)

profil3 = np.array([40, 42, 43, 40, 40, 40.5, 66, 65, 67, 65, 69, 65, 65], dtype=float)


profilGauss = np.array([40, 41, 40, 46, 61, 64, 56, 45, 41], dtype=float)


#t = np.arange(len(profilGauss))

#x0 = np.array([3, 30, 15, 1], dtype=float)

#x, flag = leastsq(residuals, x0, args=(profilGauss, t))
#print x

#Bsp Ausgleichung Gaussglocke
x0 = initialValues(profilGauss)
print 'initialValues:', x0
t = np.arange(len(profilGauss))
x, flag = leastsq(residuals, x0, args=(profilGauss, t))
print 'x: ', x, ' flag: ', flag, '\nWendestelle: ', x[2]
plt.figure()
plt.plot(t, profilGauss, 'r')
t2 = np.arange(0.0, len(profilGauss) - 0.9, 0.1)
gv = modelGauss(t2, x)
plt.plot(t2, gv, 'b')
plt.plot([x[2], x[2]], [0, modelGauss(x[2], x)], '--')

#Bsp. 1
profil1_filtered = ableitungsfilter(profil1);
print 'profil1_filtered: ', profil1_filtered
print np.argmax(profil1_filtered), ' ', max(profil1_filtered), ' ', min(profil1_filtered)
x0 = initialValues(profil1_filtered)
print 'initialValues:', x0
t = np.arange(len(profil1_filtered))
x, flag = leastsq(residuals, x0, args=(profil1_filtered, t))
print 'x: ', x, ' flag: ', flag, '\nWendestelle: ', x[2]


plt.figure()
plt.plot(t, profil1_filtered, 'r')
t2 = np.arange(0.0, len(profil1_filtered) - 0.9, 0.1)
gv = modelGauss(t2, x)
plt.plot(t2, gv, 'b')
plt.plot([x[2], x[2]], [0, modelGauss(x[2], x)], '--')

#Bsp. 2
profil2_filtered = ableitungsfilter(profil2);
print 'profil2_filtered: ', profil2_filtered
x0 = initialValues(profil2_filtered)
print 'initialValues:', x0
t = np.arange(len(profil2_filtered))
x, flag = leastsq(residuals, x0, args=(profil2_filtered, t))
print 'x: ', x, ' flag: ', flag, '\nWendestelle: ', x[2]

plt.figure()
plt.plot(t, profil2_filtered, 'r')
t2 = np.arange(0.0, len(profil2_filtered) - 0.9, 0.1)
gv = modelGauss(t2, x)
plt.plot(t2, gv, 'b')
plt.plot([x[2], x[2]], [0, modelGauss(x[2], x)], '--')


#Bsp. 3
profil3_filtered = ableitungsfilter(profil3);
print 'profil3_filtered: ', profil3_filtered
x0 = initialValues(profil3_filtered)
print 'initialValues:', x0
t = np.arange(len(profil3_filtered))
x, flag = leastsq(residuals, x0, args=(profil3_filtered, t))
print 'x: ', x, ' flag: ', flag, '\nWendestelle: ', x[2]

plt.figure()
plt.plot(t, profil3_filtered, 'r')
t2 = np.arange(0.0, len(profil3_filtered) - 0.9, 0.1)
gv = modelGauss(t2, x)
plt.plot(t2, gv, 'b')
plt.plot([x[2], x[2]], [0, modelGauss(x[2], x)], '--')

plt.show()
