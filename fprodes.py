from PyQt6 import QtCore, QtGui, QtWidgets
import joblib
import numpy as np
import re
import lime
from lime import lime_tabular
import pandas as pd
from xgboost import XGBClassifier

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setGeometry(500,200,483, 473)

        MainWindow.setStyleSheet("""
                    QWidget {
                        background-color: #121212;
                        color: #E0E0E0;
                        font-family: 'Arial';
                        font-size: 14px;
                    }
                    
                    QGroupBox {
                        background-color: transparent;
                        border: 2px solid #8B008B;
                        border-radius: 10px;
                        padding: 10px;
                        margin-top: 10px;
                    }
                    
                    QLabel {
                        color: #E0E0E0;  /* Softer white for readability */
                        font-size: 18px;  /* Bigger text for emphasis */
                        font-weight: bold;
                        padding: 8px;
                        letter-spacing: 0.5px;  /* Slight spacing for readability */
                    }
                    
                    QLabel[objectName*="lbl"] {  
                        font-size: 22px;  /* Make section headers larger */
                        color: #BB86FC;  /* Soft purple for a modern look */
                        text-transform: uppercase;  /* Make it stand out */
                    }
                    
                    QPushButton {
                        background-color: #1E88E5;
                        color: #FFFFFF;
                        font-size: 14px;
                        font-weight: bold;
                        border-radius: 8px;
                        padding: 6px;
                        border: 2px solid #1565C0;
                    }
                    
                    QPushButton:hover {
                        background-color: #1565C0;
                        border: 2px solid #0D47A1;
                        box-shadow: 0px 0px 10px #1565C0;
                    }
                    
                    QPushButton:pressed {
                        background-color: #0D47A1;
                        border: 2px solid #0B3D91;
                    }
                    
                    QRadioButton {
                        spacing: 5px;
                        font-size: 14px;
                        color: #E0E0E0;
                    }
                    
                    QRadioButton::indicator {
                        width: 14px;
                        height: 14px;
                    }
                    
                    QRadioButton::indicator::unchecked {
                        background-color: #757575;
                        border: 2px solid #9E9E9E;
                        border-radius: 7px;
                    }
                    
                    QRadioButton::indicator::checked {
                        background-color: #E53935;
                        border: 2px solid #B71C1C;
                        border-radius: 7px;
                    }
                    
                    QSpinBox {
                        background-color: #1E1E1E;
                        border: 1px solid #3A3A3A;
                        color: #E0E0E0;
                        border-radius: 5px;
                        padding: 5px;
                    }
                """)


        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.stackedWidget = QtWidgets.QStackedWidget(parent=self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 0, 481, 451))
        self.stackedWidget.setObjectName("stackedWidget")

        self.academics = QtWidgets.QWidget()
        self.academics.setObjectName("academics")
        self.grp1 = QtWidgets.QGroupBox(parent=self.academics)
        self.grp1.setObjectName('grp1')
        self.grp1.setGeometry(10, 50, 460, 300)
        self.acalbl = QtWidgets.QLabel(parent=self.academics)
        self.acalbl.setGeometry(QtCore.QRect(10, 20, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.acalbl.setFont(font)
        self.acalbl.setObjectName("acalbl")
        self.acalbl.setFixedSize(280, 40)
        self.anone = QtWidgets.QRadioButton(parent=self.academics)
        self.anone.setGeometry(QtCore.QRect(20, 70, 89, 20))
        font1 = QtGui.QFont()
        font1.setPointSize(11)
        font1.setBold(True)
        self.anone.setFont(font1)
        self.anone.setObjectName("anone")
        self.anone.setChecked(True)
        self.alow = QtWidgets.QRadioButton(parent=self.academics)
        self.alow.setGeometry(QtCore.QRect(20, 110, 89, 20))
        self.alow.setFont(font1)
        self.alow.setObjectName("alow")
        self.amoderate = QtWidgets.QRadioButton(parent=self.academics)
        self.amoderate.setGeometry(QtCore.QRect(20, 150, 91, 20))
        self.amoderate.setFont(font1)
        self.amoderate.setObjectName("amoderate")
        self.amedium = QtWidgets.QRadioButton(parent=self.academics)
        self.amedium.setGeometry(QtCore.QRect(20, 190, 89, 20))
        self.amedium.setFont(font1)
        self.amedium.setObjectName("amedium")
        self.ahigh = QtWidgets.QRadioButton(parent=self.academics)
        self.ahigh.setGeometry(QtCore.QRect(20, 230, 89, 20))
        self.ahigh.setFont(font1)
        self.ahigh.setObjectName("ahigh")
        self.aextreme = QtWidgets.QRadioButton(parent=self.academics)
        self.aextreme.setGeometry(QtCore.QRect(20, 270, 89, 20))
        self.aextreme.setFont(font1)
        self.aextreme.setObjectName("aextreme")
        self.stackedWidget.addWidget(self.academics)

        self.eating = QtWidgets.QWidget()
        self.eating.setObjectName("eating")
        self.grp2 = QtWidgets.QGroupBox(parent=self.eating)
        self.grp2.setObjectName('grp2')
        self.grp2.setGeometry(10, 50, 460, 300)
        self.eatlbl = QtWidgets.QLabel(parent=self.eating)
        self.eatlbl.setGeometry(QtCore.QRect(10, 20, 151, 31))
        self.eatlbl.setFont(font)
        self.eatlbl.setObjectName("eatlbl")
        self.eunhealthy = QtWidgets.QRadioButton(parent=self.eating)
        self.eunhealthy.setGeometry(QtCore.QRect(20, 70, 101, 20))
        self.eunhealthy.setFont(font1)
        self.eunhealthy.setObjectName("eunhealthy")
        self.ehealthy = QtWidgets.QRadioButton(parent=self.eating)
        self.ehealthy.setGeometry(QtCore.QRect(20, 150, 89, 20))
        self.ehealthy.setFont(font1)
        self.ehealthy.setObjectName("ehealthy")
        self.emoderate = QtWidgets.QRadioButton(parent=self.eating)
        self.emoderate.setGeometry(QtCore.QRect(20, 110, 180, 20))
        self.emoderate.setFont(font1)
        self.emoderate.setObjectName("emoderate")
        self.emoderate.setChecked(True)
        self.stackedWidget.addWidget(self.eating)

        self.finance = QtWidgets.QWidget()
        self.finance.setObjectName("finance")
        self.grp3 = QtWidgets.QGroupBox(parent=self.finance)
        self.grp3.setObjectName('grp3')
        self.grp3.setGeometry(10, 50, 460, 300)
        self.finlbl = QtWidgets.QLabel(parent=self.finance)
        self.finlbl.setGeometry(QtCore.QRect(10, 20, 211, 31))
        self.finlbl.setFont(font)
        self.finlbl.setObjectName("finlbl")
        self.fnone = QtWidgets.QRadioButton(parent=self.finance)
        self.fnone.setGeometry(QtCore.QRect(20, 70, 89, 20))
        self.fnone.setFont(font1)
        self.fnone.setObjectName("fnone")
        self.fnone.setChecked(True)
        self.flow = QtWidgets.QRadioButton(parent=self.finance)
        self.flow.setGeometry(QtCore.QRect(20, 110, 89, 20))
        self.flow.setFont(font1)
        self.flow.setObjectName("flow")
        self.fmoderate = QtWidgets.QRadioButton(parent=self.finance)
        self.fmoderate.setGeometry(QtCore.QRect(20, 150, 101, 20))
        self.fmoderate.setFont(font1)
        self.fmoderate.setObjectName("fmoderate")
        self.fmedium = QtWidgets.QRadioButton(parent=self.finance)
        self.fmedium.setGeometry(QtCore.QRect(20, 190, 89, 20))
        self.fmedium.setFont(font1)
        self.fmedium.setObjectName("fmedium")
        self.fhigh = QtWidgets.QRadioButton(parent=self.finance)
        self.fhigh.setGeometry(QtCore.QRect(20, 230, 89, 20))
        self.fhigh.setFont(font1)
        self.fhigh.setObjectName("fhigh")
        self.fextreme = QtWidgets.QRadioButton(parent=self.finance)
        self.fextreme.setGeometry(QtCore.QRect(20, 270, 89, 20))
        self.fextreme.setFont(font1)
        self.fextreme.setObjectName("fextreme")
        self.stackedWidget.addWidget(self.finance)

        self.suicide = QtWidgets.QWidget()
        self.suicide.setObjectName("suicide")
        self.grp4 = QtWidgets.QGroupBox(parent=self.suicide)
        self.grp4.setObjectName('grp4')
        self.grp4.setGeometry(10, 50, 460, 300)
        self.suilbl = QtWidgets.QLabel(parent=self.suicide)
        self.suilbl.setGeometry(QtCore.QRect(10, 20, 281, 31))
        self.suilbl.setFont(font)
        self.suilbl.setObjectName("suilbl")
        self.syes = QtWidgets.QRadioButton(parent=self.suicide)
        self.syes.setGeometry(QtCore.QRect(20, 70, 89, 20))
        self.syes.setFont(font1)
        self.syes.setObjectName("syes")
        self.sno = QtWidgets.QRadioButton(parent=self.suicide)
        self.sno.setGeometry(QtCore.QRect(20, 110, 89, 20))
        self.sno.setFont(font1)
        self.sno.setObjectName("sno")
        self.sno.setChecked(True)
        self.stackedWidget.addWidget(self.suicide)

        self.employment = QtWidgets.QWidget()
        self.employment.setObjectName("employment")
        self.grp5 = QtWidgets.QGroupBox(parent=self.employment)
        self.grp5.setObjectName('grp5')
        self.grp5.setGeometry(10, 50, 460, 300)
        self.emplbl = QtWidgets.QLabel(parent=self.employment)
        self.emplbl.setGeometry(QtCore.QRect(10, 20, 281, 34))
        self.emplbl.setFont(font)
        self.emplbl.setObjectName("emplbl")
        self.empyes = QtWidgets.QRadioButton(parent=self.employment)
        self.empyes.setGeometry(QtCore.QRect(20, 70, 89, 20))
        self.empyes.setFont(font1)
        self.empyes.setObjectName("empyes")
        self.empno = QtWidgets.QRadioButton(parent=self.employment)
        self.empno.setGeometry(QtCore.QRect(20, 110, 120, 20))
        self.empno.setFont(font1)
        self.empno.setObjectName("empno")
        self.empno.setChecked(True)
        self.stackedWidget.addWidget(self.employment)

        self.illness = QtWidgets.QWidget()
        self.illness.setObjectName("illness")
        self.grp6 = QtWidgets.QGroupBox(parent=self.illness)
        self.grp6.setObjectName('grp6')
        self.grp6.setGeometry(10, 80, 460, 300)
        self.illlbl = QtWidgets.QLabel(parent=self.illness)
        self.illlbl.setGeometry(QtCore.QRect(10, 17, 500, 34))
        self.illlbl.setFont(font)
        self.illlbl.setObjectName("illlbl")
        self.illlbl1 = QtWidgets.QLabel(parent=self.illness)
        self.illlbl1.setGeometry(QtCore.QRect(10, 47, 500, 34))
        self.illlbl1.setFont(font1)
        self.illlbl1.setObjectName("illlbl1")
        self.illyes = QtWidgets.QRadioButton(parent=self.illness)
        self.illyes.setGeometry(QtCore.QRect(20, 100, 89, 20))
        self.illyes.setFont(font1)
        self.illyes.setObjectName("illyes")
        self.illno = QtWidgets.QRadioButton(parent=self.illness)
        self.illno.setGeometry(QtCore.QRect(20, 140, 89, 20))
        self.illno.setFont(font1)
        self.illno.setObjectName("illno")
        self.illno.setChecked(True)
        self.stackedWidget.addWidget(self.illness)

        self.studydata = QtWidgets.QWidget()
        self.studydata.setObjectName("studydata")
        self.grp7 = QtWidgets.QGroupBox(parent=self.studydata)
        self.grp7.setObjectName('grp7')
        self.grp7.setGeometry(10, 5, 460, 120)
        self.sdylbl = QtWidgets.QLabel(parent=self.studydata)
        self.sdylbl.setGeometry(QtCore.QRect(20, 20, 301, 31))
        self.sdylbl.setFont(font)
        self.sdylbl.setObjectName("sdylbl")
        self.hstudy = QtWidgets.QSpinBox(parent=self.studydata)
        self.hstudy.setGeometry(QtCore.QRect(40, 70, 81, 31))
        self.hstudy.setObjectName("hstudy")
        self.hstudy.setRange(0, 18)
        self.grp8 = QtWidgets.QGroupBox(parent=self.studydata)
        self.grp8.setObjectName('grp8')
        self.grp8.setGeometry(10, 180, 460, 120)
        self.agelbl = QtWidgets.QLabel(parent=self.studydata)
        self.agelbl.setGeometry(QtCore.QRect(20, 200, 171, 31))
        self.agelbl.setFont(font)
        self.agelbl.setObjectName("agelbl")
        self.sage = QtWidgets.QSpinBox(parent=self.studydata)
        self.sage.setGeometry(QtCore.QRect(40, 250, 81, 31))
        self.sage.setObjectName("sage")
        self.sage.setRange(18, 60)
        self.sage.setValue(18)
        self.stackedWidget.addWidget(self.studydata)

        MainWindow.setCentralWidget(self.centralwidget)
        self.acagrp = QtWidgets.QButtonGroup(self.centralwidget)
        self.acagrp.setObjectName('acagrp')
        self.acagrp.addButton(self.anone)
        self.acagrp.addButton(self.alow)
        self.acagrp.addButton(self.amoderate)
        self.acagrp.addButton(self.amedium)
        self.acagrp.addButton(self.ahigh)
        self.acagrp.addButton(self.aextreme)

        self.eatgrp = QtWidgets.QButtonGroup(self.centralwidget)
        self.eatgrp.setObjectName("eatgrp")
        self.eatgrp.addButton(self.ehealthy)
        self.eatgrp.addButton(self.eunhealthy)
        self.eatgrp.addButton(self.emoderate)

        self.suigrp = QtWidgets.QButtonGroup(self.centralwidget)
        self.suigrp.setObjectName('suigrp')
        self.suigrp.addButton(self.syes)
        self.suigrp.addButton(self.sno)

        self.fingrp = QtWidgets.QButtonGroup(self.centralwidget)
        self.fingrp.setObjectName('fingrp')
        self.fingrp.addButton(self.fnone)
        self.fingrp.addButton(self.flow)
        self.fingrp.addButton(self.fmoderate)
        self.fingrp.addButton(self.fmedium)
        self.fingrp.addButton(self.fhigh)
        self.fingrp.addButton(self.fextreme)

        self.empgrp = QtWidgets.QButtonGroup(self.centralwidget)
        self.empgrp.setObjectName('empgrp')
        self.empgrp.addButton(self.empyes)
        self.empgrp.addButton(self.empno)

        self.illgrp = QtWidgets.QButtonGroup(self.centralwidget)
        self.illgrp.setObjectName('illgrp')
        self.illgrp.addButton(self.illyes)
        self.illgrp.addButton(self.illno)

        self.nextButton1 = QtWidgets.QPushButton("Next", self.academics)
        self.nextButton1.setGeometry(QtCore.QRect(350, 400, 100, 30))
        self.nextButton1.clicked.connect(lambda: self.nextPage(1))

        self.nextButton2 = QtWidgets.QPushButton("Next", self.eating)
        self.nextButton2.setGeometry(QtCore.QRect(350, 400, 100, 30))
        self.nextButton2.clicked.connect(lambda: self.nextPage(2))

        self.nextButton3 = QtWidgets.QPushButton("Next", self.finance)
        self.nextButton3.setGeometry(QtCore.QRect(350, 400, 100, 30))
        self.nextButton3.clicked.connect(lambda: self.nextPage(3))

        self.nextButton4 = QtWidgets.QPushButton("Next", self.suicide)
        self.nextButton4.setGeometry(QtCore.QRect(350, 400, 100, 30))
        self.nextButton4.clicked.connect(lambda: self.nextPage(4))

        self.nextButton5 = QtWidgets.QPushButton("Next", self.employment)
        self.nextButton5.setGeometry(QtCore.QRect(350, 400, 100, 30))
        self.nextButton5.clicked.connect(lambda: self.nextPage(5))

        self.nextButton6 = QtWidgets.QPushButton("Next", self.illness)
        self.nextButton6.setGeometry(QtCore.QRect(350, 400, 100, 30))
        self.nextButton6.clicked.connect(lambda: self.nextPage(6))

        self.submit = QtWidgets.QPushButton("Submit", self.studydata)
        self.submit.setGeometry(QtCore.QRect(350, 400, 100, 30))
        self.submit.clicked.connect(lambda: self.get_inputs())

        self.previousButton5 = QtWidgets.QPushButton(parent=self.eating)
        self.previousButton5.setGeometry(QtCore.QRect(20, 400, 100, 30))
        self.previousButton5.setText("Previous")
        self.previousButton5.clicked.connect(self.previousPage)

        self.previousButton4 = QtWidgets.QPushButton(parent=self.employment)
        self.previousButton4.setGeometry(QtCore.QRect(20, 400, 100, 30))
        self.previousButton4.setText("Previous")
        self.previousButton4.clicked.connect(self.previousPage)

        self.previousButton1 = QtWidgets.QPushButton(parent=self.finance)
        self.previousButton1.setGeometry(QtCore.QRect(20, 400, 100, 30))
        self.previousButton1.setText("Previous")
        self.previousButton1.clicked.connect(self.previousPage)

        self.previousButton2 = QtWidgets.QPushButton(parent=self.studydata)
        self.previousButton2.setGeometry(QtCore.QRect(20, 400, 100, 30))
        self.previousButton2.setText("Previous")
        self.previousButton2.clicked.connect(self.previousPage)

        self.previousButton3 = QtWidgets.QPushButton(parent=self.suicide)
        self.previousButton3.setGeometry(QtCore.QRect(20, 400, 100, 30))
        self.previousButton3.setText("Previous")
        self.previousButton3.clicked.connect(self.previousPage)

        self.previousButton6 = QtWidgets.QPushButton(parent=self.illness)
        self.previousButton6.setGeometry(QtCore.QRect(20, 400, 100, 30))
        self.previousButton6.setText("Previous")
        self.previousButton6.clicked.connect(self.previousPage)

        self.acalbl.setFixedSize(350, 40)
        self.finlbl.setFixedSize(350, 40)
        self.eatlbl.setFixedSize(300, 40)
        self.suilbl.setFixedSize(450, 40)
        self.sdylbl.setFixedSize(430, 40)
        self.agelbl.setFixedSize(350, 40)


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def get_inputs(self):
        saca, semp, ssui, seat, sfin, sill = None, None, None, None, None, None
        selected_academics = self.acagrp.checkedButton()
        if selected_academics:
            saca = selected_academics.text()

        selected_employment = self.empgrp.checkedButton()
        if selected_employment:
            semp = selected_employment.text()

        selected_eating = self.eatgrp.checkedButton()
        if selected_eating:
            seat = selected_eating.text()

        selected_finance = self.fingrp.checkedButton()
        if selected_finance:
            sfin = selected_finance.text()

        selected_suicide = self.suigrp.checkedButton()
        if selected_suicide:
            ssui = selected_suicide.text()

        selected_illness = self.illgrp.checkedButton()
        if selected_illness:
            sill = selected_illness.text()

        age = self.sage.value()
        shours = self.hstudy.value()
        valdict = {'None': 0, 'Low': 1, 'Moderate': 2, 'Medium': 3, 'High': 4, 'Extreme': 5, 'Healthy': [0, 0], 'Employed': 1, 'Unemployed': 0, 'Moderately Healthy': [1, 0], 'Unhealthy': [0, 1], 'Yes': 1, 'No': 0}
        inpvals = [age, valdict[saca], shours, valdict[sfin], valdict[seat], valdict[ssui], valdict[sill], valdict[semp]]
        flattened = [item for sublist in inpvals for item in (sublist if isinstance(sublist, list) else [sublist])]
        model = joblib.load('xboostdepro.pkl')
        s = None
        ans = 'Depressed' if model.predict(np.array(flattened).reshape(1, -1))[0] == 1 else 'Not Depressed'
        if ans == 'Depressed':
            result_text = "You are clinically Depressed"
            s = "background-color: #4A148C; color: white;"
        else:
            result_text = "You are not Depressed"
            s = "background-color: #1E88E5; color: black;"
        advice = None
        if ans == 'Depressed':
            x_train = pd.read_csv('dxtrain').drop(columns='Unnamed: 0')
            explainer = lime.lime_tabular.LimeTabularExplainer(
                x_train.values,
                feature_names=x_train.columns,
                class_names=['Not Depressed', 'Depressed'],
                mode="classification"
            )

            exp = explainer.explain_instance(np.array(flattened), model.predict_proba)

            pack = [i for i in exp.as_list() if i[1] >= 0.1]
            namepack = [re.sub(r"[^a-zA-Z\s]", "", i[0]).strip().lower() for i in exp.as_list() if i[1] >= 0.1]
            valpack = [i[1] for i in exp.as_list() if i[1] >= 0.1]

            imps = {i: j for i, j in zip(namepack, valpack)}

            words = {'sui': 'You should consider therapy', 'aca': 'Optimize your study habits',
                     'die': 'Consume healthier food options', 'study': 'adjust your work/study time',
                     'fin': 'Seek financial advice from knowns or professionals'}
            advice = []
            for i in imps.keys():
                for j in words.keys():
                    if j in i:
                        advice.append(words[j])
        self.result_window = ResultDialog(result_text, advice)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Depression Test"))
        self.acalbl.setText(_translate("MainWindow", "Academic Pressure"))
        self.anone.setText(_translate("MainWindow", "None"))
        self.alow.setText(_translate("MainWindow", "Low"))
        self.amoderate.setText(_translate("MainWindow", "Moderate"))
        self.amedium.setText(_translate("MainWindow", "Medium"))
        self.ahigh.setText(_translate("MainWindow", "High"))
        self.aextreme.setText(_translate("MainWindow", "Extreme"))
        self.eatlbl.setText(_translate("MainWindow", "Eating Habits"))
        self.eunhealthy.setText(_translate("MainWindow", "Unhealthy"))
        self.ehealthy.setText(_translate("MainWindow", "Healthy"))
        self.emoderate.setText(_translate("MainWindow", "Moderately Healthy"))
        self.finlbl.setText(_translate("MainWindow", "Financial Stress level"))
        self.fnone.setText(_translate("MainWindow", "None"))
        self.flow.setText(_translate("MainWindow", "Low"))
        self.fmoderate.setText(_translate("MainWindow", "Moderate"))
        self.fmedium.setText(_translate("MainWindow", "Medium"))
        self.fhigh.setText(_translate("MainWindow", "High"))
        self.fextreme.setText(_translate("MainWindow", "Extreme"))
        self.suilbl.setText(_translate("MainWindow", "Have you ever felt suicidal"))
        self.syes.setText(_translate("MainWindow", "Yes"))
        self.sno.setText(_translate("MainWindow", "No"))
        self.emplbl.setText(_translate('MainWindow', 'Employment Status'))
        self.empyes.setText(_translate('MainWindow','Employed'))
        self.empno.setText(_translate('MainWindow', 'Unemployed'))
        self.illlbl.setText(_translate("MainWindow", 'Does your family have a Mental'))
        self.illlbl1.setText(_translate("MainWindow", 'Illness history'))
        self.illyes.setText(_translate("MainWindow", "Yes"))
        self.illno.setText(_translate("MainWindow", "No"))
        self.sdylbl.setText(_translate("MainWindow", "Enter hours studied everyday"))
        self.agelbl.setText(_translate("MainWindow", "Enter your Age"))


    def nextPage(self, index):
        self.stackedWidget.setCurrentIndex(index)

    def previousPage(self):
        current_index = self.stackedWidget.currentIndex()
        if current_index > 0:
            self.stackedWidget.setCurrentIndex(current_index - 1)

class ResultDialog(QtWidgets.QDialog):
    def __init__(self, result, advice=None):
        super().__init__()
        self.setWindowTitle("Prediction Result")
        self.setFixedSize(400, 250)

        layout = QtWidgets.QVBoxLayout()
        result_color = "#1E88E5" if 'not' in result else "#E53935"
        result_label = QtWidgets.QLabel(result)
        result_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        result_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {result_color};")
        layout.addWidget(result_label)

        if advice:
            self.advices = {}
            for i, a in enumerate(advice):
                self.advices[f"advice_label{i}"] = QtWidgets.QLabel(a)
                self.advices[f"advice_label{i}"].setStyleSheet("font-size: 14px; color: #BBBBBB;")
                layout.addWidget(self.advices[f"advice_label{i}"])

        close_button = QtWidgets.QPushButton("OK")
        close_button.setStyleSheet("background-color: #1E88E5; color: white; border-radius: 6px; padding: 6px;")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        self.setLayout(layout)
        self.show()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
