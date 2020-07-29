#Created by Sawan

from pandas import DataFrame
from sklearn import linear_model
import tkinter as tk
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

import math




Data = {
     'Exercise(in hr) pr week': [18,8,6,7,12,7,4,3,5,5,0,0,0,0,12,0,0,14,14,14,0,8,14,8,16,7,9,4,13,5],
    'Weight': [65,60,75,70,72,69,54,80,68,60,65,88,85,80,75,85,70,60,86,85,92,60,70,68,68,88,70,75,71,74],
    'Height': [168,155,174,168,174,168,161,177,180,177,164,180,177,171,180,174,158,174,183,168,183,171,189,177,180,177,174,180,189,178]
}

df = DataFrame(Data, columns=['Exercise(in hr) pr week', 'Weight','Height'])

X = df[['Exercise(in hr) pr week',
        'Height']]  # here we have 2 input variables for multiple regression.
Y = df['Weight']  # output variable (what we are trying to predict)
Z= df['Exercise(in hr) pr week']

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
X = sm.add_constant(X)  # adding a constant
model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

# tkinter GUI
root = tk.Tk()
canvas1 = tk.Canvas(root, width=750, height=450)
canvas1.pack()




# with sklearn
interceptfordisplay=76.45471405321526
Intercept_result = ('Intercept: ', interceptfordisplay)
label_Intercept = tk.Label(root, text=Intercept_result, justify='center')
canvas1.create_window(410, 220, window=label_Intercept)

# with sklearn
Coefficients_result = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify='center')
canvas1.create_window(410, 240, window=label_Coefficients)

a = [18,8,6,7,12,7,4,3,5,5,0,0,0,0,12,0,0,14,14,14,0,8,14,8,16,7,9,4,13,5]
b = [65,60,75,70,72,69,54,80,68,60,65,88,85,80,75,85,70,60,86,85,92,60,70,68,68,88,70,75,71,74]
slope=linregress(a, b)
slopefordisplay=-0.49596911547632566
Slopeview= ('slope: ', slopefordisplay)
label_line = tk.Label(root, text=Slopeview, justify='center')
canvas1.create_window(410, 260, window=label_line)
print(slope)

Equation = ('y = -0.49596911547632566 x + 76.45471405321526 ')
label_equation = tk.Label(root, text=Equation, justify='center')
canvas1.create_window(410, 280, window=label_equation)

labelMain = tk.Label(root, text='Probability Calculator: Stats Project')
canvas1.create_window(400, 50, window=labelMain)


# Excercise label and input box
label1 = tk.Label(root, text='Excercise(in hour) per week: ')
canvas1.create_window(250, 100, window=label1)

entry1 = tk.Entry(root)  # create 1st entry box
canvas1.create_window(420, 100, window=entry1)

# Height label and input box
label2 = tk.Label(root, text='height(in cm): ')
canvas1.create_window(270, 120, window=label2)

entry2 = tk.Entry(root)  # create 2nd entry box
canvas1.create_window(420, 120, window=entry2)


labelchivalue = tk.Label(root, text='Chi-value: ')
canvas1.create_window(50, 340, window=labelchivalue)

labelchisquare = tk.Label(root, text='Chi-square Distribution Calculator ')
canvas1.create_window(150, 440, window=labelchisquare)

chivalue= tk.Entry(root)
canvas1.create_window(150, 340, window=chivalue)

labeldegreeoffreedom = tk.Label(root, text='degree: ')
canvas1.create_window(50, 360, window=labeldegreeoffreedom)
degreeoffreedom= tk.Entry(root)
canvas1.create_window(150, 360, window=degreeoffreedom)

def prob():
    chi_Value = float(chivalue.get())
    degrees_of_Freedom = int(degreeoffreedom.get())
    Delta_x = ((chi_Value - 0) / 1800)
    xValArr = []
    i = 0
    for p in range(1800 + 1):
        xValArr.append(0 + (i * Delta_x))
        i += 1
    xValArr1 = []
    for q in range(len(xValArr)):
        xValArr1.append(pow(xValArr[q], (degrees_of_Freedom / 2) - 1) * math.exp((-1 * xValArr[q]) / 2))
    xValArr_final = []
    for ff in range(len(xValArr1)):
        xValArr_final.append(xValArr1[ff] / (pow(2, degrees_of_Freedom / 2) * math.gamma(degrees_of_Freedom / 2)))
    xValArr_final_first = xValArr_final[0]
    xValArr_final_last = xValArr_final[len(xValArr_final) - 1]
    xValArr_EvenSum = 0
    xValArr_OddSum = 0
    for j in range(len(xValArr_final)):
        if j % 2 == 0:
            xValArr_EvenSum = xValArr_EvenSum + xValArr_final[j]
        else:
            xValArr_OddSum = xValArr_OddSum + xValArr_final[j]
    xValArr_OddSum = xValArr_OddSum - xValArr_final_first
    xValArr_EvenSum = xValArr_EvenSum - xValArr_final_last
    result = (Delta_x / 3) * (xValArr_final_first + (4 * xValArr_OddSum) + (2 * xValArr_EvenSum) + xValArr_final_last)

    label_result = tk.Label(root, text=result, bg='orange')
    canvas1.create_window(150, 410, window=label_result)



button3 = tk.Button(root, text='Probability', command=prob,
                    bg='red')  # button to generate the graph
canvas1.create_window(150, 390, window=button3)



labelzscore = tk.Label(root, text='z-score: ')
canvas1.create_window(290, 340, window=labelzscore)
zscore= tk.Entry(root)
canvas1.create_window(380, 340, window=zscore)

labelnormal = tk.Label(root, text='Normal Distribution Calculator ')
canvas1.create_window(380, 440, window=labelnormal)


def prob2():
    z_Score = float(zscore.get())
    Delta_x = ((z_Score - 0) / 1800)
    xValArr = []
    i = 0
    denom = 1 / (math.sqrt(2 * math.pi))
    for p in range(1800 + 1):
        xValArr.append(0 + (i * Delta_x))
        i += 1
    xValArr1 = []
    for q in range(len(xValArr)):
        xValArr1.append(math.exp(((-1 / 2) * pow(xValArr[q], 2))) * denom)
    xValArr_final_first = xValArr1[0]
    xValArr_final_last = xValArr1[len(xValArr1) - 1]
    xValArr_EvenSum = 0
    xValArr_OddSum = 0
    for r in range(len(xValArr1)):
        if r % 2 == 0:
            xValArr_EvenSum = xValArr_EvenSum + xValArr1[r]
        else:
            xValArr_OddSum = xValArr_OddSum + xValArr1[r]
    xValArr_OddSum = xValArr_OddSum - xValArr_final_first
    xValArr_EvenSum = xValArr_EvenSum - xValArr_final_last
    Prob = ((Delta_x / 3) * (xValArr_final_first + (4 * xValArr_OddSum) + (2 * (xValArr_EvenSum)) + xValArr_final_last))
    def checkZscore():
        if z_Score > 0:
            return 1
        elif z_Score == 0:
            return 0
        else:
            return -1
    VAL = checkZscore()
    final_Prob3 = 0
    if VAL == 1:
        final_Prob3 = Prob + 0.5
    elif VAL == -1:
        final_Prob3 = Prob - 0.5
    else:
        final_Prob3 = 0

    label_result2 = tk.Label(root, text=final_Prob3, bg='orange')
    canvas1.create_window(380, 410, window=label_result2)

button4 = tk.Button(root, text='Probability', command=prob2,
                    bg='red')  # button to generate the graph
canvas1.create_window(380, 390, window=button4)


labeltvalue = tk.Label(root, text='T-value: ')
canvas1.create_window(520, 340, window=labeltvalue)
tvalue= tk.Entry(root)
canvas1.create_window(610, 340, window=tvalue)

labeldegreeoffreedom2 = tk.Label(root, text='degree: ')
canvas1.create_window(520, 360, window=labeldegreeoffreedom2)
degreeoffreedom2= tk.Entry(root)
canvas1.create_window(610, 360, window=degreeoffreedom2)

labelt = tk.Label(root, text='T-Distribution Calculator ')
canvas1.create_window(610, 440, window=labelt)

def prob3():
    t_Value = float(tvalue.get())
    degrees_of_Freedom = int(degreeoffreedom2.get())
    Delta_x = ((t_Value - 0) / 1800)
    xValArr = []
    i = 0
    const = (math.gamma((degrees_of_Freedom + 1) / 2)) / (
            math.sqrt(degrees_of_Freedom * 3.14) * math.gamma(degrees_of_Freedom / 2))
    power = (-1 / 2) * (degrees_of_Freedom + 1)
    for p in range(1800 + 1):
        xValArr.append(0 + (i * Delta_x))
        i += 1
    xValArr1 = []
    for q in range(len(xValArr)):
        xValArr1.append((pow((1 + (pow(xValArr[q], 2) / degrees_of_Freedom)), power)) * const)
    xValArr_final_first = xValArr1[0]
    xValArr_final_last = xValArr1[len(xValArr1) - 1]
    xValArr_EvenSum = 0
    xValArr_OddSum = 0
    for j in range(len(xValArr1)):
        if j % 2 == 0:
            xValArr_EvenSum = xValArr_EvenSum + xValArr1[j]
        else:
            xValArr_OddSum = xValArr_OddSum + xValArr1[j]
    xValArr_OddSum = xValArr_OddSum - xValArr_final_first
    xValArr_EvenSum = xValArr_EvenSum - xValArr_final_last
    Prob = (Delta_x / 3) * (xValArr_final_first + (4 * xValArr_OddSum) + (2 * xValArr_EvenSum) + xValArr_final_last)
    def checkTvalue():
        if t_Value > 0:
            return 1
        elif t_Value == 0:
            return 0
        else:
            return -1
    VAL = checkTvalue()
    final_Prob2 = 0
    if VAL == 1:
        final_Prob2 = Prob + 0.5
    elif VAL == -1:
        final_Prob2 = Prob - 0.5
    else:
        final_Prob2 = 0

    label_result3 = tk.Label(root, text=final_Prob2, bg='orange')
    canvas1.create_window(610, 410, window=label_result3)



button5 = tk.Button(root, text='Probability', command=prob3,
                    bg='red')  # button to generate the graph
canvas1.create_window(610, 390, window=button5)


def values():
    global New_Excercise  # our 1st input variable
    New_Excercise  = float(entry1.get())
    global New_Height # our 2nd input variable
    New_Height = float(entry2.get())
    Prediction_result = ('Predicted Weight: ', regr.predict([[New_Excercise , New_Height]]))
    label_Prediction = tk.Label(root, text=Prediction_result, bg='orange')
    canvas1.create_window(410, 310, window=label_Prediction)

def graph():
    plt.scatter(df['Exercise(in hr) pr week'],df['Weight'], df['Height'], color='red')
    fit = np.polyfit(Z, Y, 1, cov=True)  # linear
    best_fit_y = fit[0][0] * Z+ fit[0][1]  # first index takes fit params, second specifies
    plt.plot(Z, best_fit_y, '--');
    plt.title('Excercise vs Weight', fontsize=14)
    plt.xlabel('Excercise', fontsize=14)
    plt.ylabel('Weight', fontsize=14)
    plt.grid(True)
    plt.show()

button1 = tk.Button(root, text='Predicted Weight', command=values,
                    bg='orange')  # button to call the 'values' command above
canvas1.create_window(420, 150, window=button1)

button2 = tk.Button(root, text='Graph', command=graph,
                    bg='red')  # button to generate the graph
canvas1.create_window(420, 180, window=button2)

root.mainloop()

#created by sawan