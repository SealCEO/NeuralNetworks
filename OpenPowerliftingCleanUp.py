import pandas as pd

rawData = pd.read_csv("openpowerlifting.csv", dtype = str)
rawData = rawData.drop(['Squat1Kg','Squat2Kg','Squat3Kg','Squat4Kg','Bench1Kg','Bench2Kg','Bench3Kg','Bench4Kg','Deadlift1Kg','Deadlift2Kg','Deadlift3Kg','Deadlift4Kg'], axis = 1)
rawData = rawData.drop(['IPFPoints','Country','Federation','Date','MeetCountry','MeetState','MeetName'], axis = 1)
rawData = rawData.drop(['Name','AgeClass','Division','WeightClassKg'], axis = 1)
rawData['Equipment'].replace({'Straps': None, 'Raw': 0, 'Wraps': 1, 'Single-ply': 2, 'Multi-ply': 3}, inplace = True)
rawData['Sex'].replace({'F': 0, 'M': 1}, inplace = True)
rawData['Tested'].replace({'No': 0, 'Yes': 1}, inplace = True)
rawData['Place'].replace({'DQ': None, 'G': None}, inplace = True)
rawData = rawData.dropna()
rawData = rawData.drop(['TotalKg', 'Event'], axis = 1)
def placing4AndUpToZero(x):
    x = int(x)
    if x >= 4:
        return 0
    return x
rawData['Place'] = rawData['Place'].apply(placing4AndUpToZero)

# __________Don't implement________
# breakIntoBinary = False
# #This will break the variable age and weight into binary variables
# #This has better accuracy for LDA
# #This will also combine sex and weightclasses
# #Set breakIntoBinary = True to run this part
# #This also requires a lot of computing power
# #
# #For example
# #
# #breakIntoBinary = True
# #The accuracy score for for LDA: 0.4546790269320488
# #The accuracy score for Naive Bayes:  0.3847435547880881
# #The accuracy score for KNN where k = 5 is:  0.39790663152473565
# #
# #breakIntoBinary = False
# #The accuracy score for for LDA: 0.4469387136841467
# #The accuracy score for Naive Bayes: 0.4345784483019782
# #The accuracy score for KNN where k = 5 is: 0.4061165136780878

# if breakIntoBinary:
#     def get_age_group(age):
#         age = float(age)
#         if age < 15:
#             return 1
#         elif age == 16 or age == 17:
#             return 2
#         elif age == 18 and age == 19:
#             return 3
#         elif age >= 20 and age <= 23:
#             return 4
#         elif age >= 24 and age < 40:
#             return 5
#         elif age >= 50 and age < 45:
#             return 6
#         elif age >= 45 and age < 55:
#             return 7
#         else:
#             return 8

#     def get_weight_group_by_sex(weight, sex):
#         weight = float(weight)
#         sex = int(sex)
#         if sex == 1:
#             #Male weightclasses by USPA in kg
#             if weight <= 52:
#                 return 1
#             if weight <= 56:
#                 return 2
#             if weight <= 60:
#                 return 3
#             if weight <= 67.5:
#                 return 4
#             if weight <= 75:
#                 return 5
#             if weight <= 82.5:
#                 return 6
#             if weight <= 90:
#                 return 7
#             if weight <= 100:
#                 return 8
#             if weight <= 110:
#                 return 9
#             if weight <= 125:
#                 return 10
#             if weight <= 140:
#                 return 11
#             return 12
#         else:
#             #Female weightclasses by USPA in kg
#             if weight <= 44:
#                 return 13
#             if weight <= 48:
#                 return 14
#             if weight <= 52:
#                 return 15
#             if weight <= 56:
#                 return 16
#             if weight <= 60:
#                 return 17
#             if weight <= 67.5:
#                 return 18
#             if weight <= 75:
#                 return 19
#             if weight <= 82.5:
#                 return 20
#             if weight <= 90:
#                 return 21
#             if weight <= 100:
#                 return 22
#             if weight <= 110:
#                 return 23
#             return 24
    
#     def returnSexAsString(sex):
#         sex = int(sex)
#         if sex == 0:
#             return 'F'
#         return 'M'
    
#     sexDic = {0: 'F', 1: 'M'}
#     for i in range(1, 9):
#         rawData['AgeDivision' + str(i)] = (rawData['Age'].apply(get_age_group) == i)*1
#     for i in range(1, 13):
#         rawData['WeightClass' + str(i) + 'M'] = (rawData.apply(lambda row: get_weight_group_by_sex(row['BodyweightKg'], row['Sex']), axis = 1) == i)*1
#     for i in range(1, 13):
#         rawData['WeightClass' + str(i) + 'F'] = (rawData.apply(lambda row: get_weight_group_by_sex(row['BodyweightKg'], row['Sex']), axis = 1) == (i + 12))*1
#     rawData = rawData.drop(['Age', 'BodyweightKg', 'Sex'], axis = 1)


f = open("CleanedUpPowerlifting.csv", "w")
f.truncate()
f.close()
rawData.to_csv('CleanedUpPowerlifting.csv', index = False)
