import os
import sys
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

CALCE_train = ['CALCE_CS2_36.pkl', 'CALCE_CS2_37.pkl', 'CALCE_CX2_33.pkl', 'CALCE_CS2_34.pkl', 'CALCE_CX2_37.pkl', 'CALCE_CS2_33.pkl', 'CALCE_CS2_35.pkl', 'CALCE_CS2_38.pkl', 'CALCE_CX2_36.pkl']
CALCE_vali = ['CALCE_CX2_35.pkl', 'CALCE_CX2_34.pkl']
CALCE_test = ['CALCE_CX2_16.pkl', 'CALCE_CX2_38.pkl']

HNEI_train =  ['HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_o.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_p.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_e.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_l.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_b.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_t.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_c.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_a.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_g.pkl']
HNEI_vali = ['HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_n.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_s.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_j.pkl']
HNEI_test =  ['HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_f.pkl', 'HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_d.pkl']

HUST_train = ['HUST_1-6.pkl', 'HUST_2-2.pkl', 'HUST_1-3.pkl', 'HUST_6-3.pkl', 'HUST_1-2.pkl', 'HUST_3-7.pkl', 'HUST_3-2.pkl', 'HUST_10-6.pkl', 'HUST_3-6.pkl', 'HUST_5-1.pkl', 'HUST_10-5.pkl', 'HUST_6-2.pkl', 'HUST_6-1.pkl', 'HUST_8-1.pkl', 'HUST_10-7.pkl', 'HUST_1-4.pkl', 'HUST_5-4.pkl', 'HUST_1-5.pkl', 'HUST_6-6.pkl', 'HUST_5-6.pkl', 'HUST_6-4.pkl', 'HUST_9-2.pkl', 'HUST_10-4.pkl', 'HUST_5-3.pkl', 'HUST_7-7.pkl', 'HUST_3-1.pkl', 'HUST_4-1.pkl', 'HUST_4-4.pkl', 'HUST_4-6.pkl', 'HUST_8-8.pkl', 'HUST_2-4.pkl', 'HUST_9-8.pkl', 'HUST_9-5.pkl', 'HUST_3-3.pkl', 'HUST_1-7.pkl', 'HUST_4-5.pkl', 'HUST_9-6.pkl', 'HUST_1-1.pkl', 'HUST_4-3.pkl', 'HUST_2-5.pkl', 'HUST_4-7.pkl', 'HUST_7-2.pkl', 'HUST_8-4.pkl', 'HUST_3-5.pkl', 'HUST_2-6.pkl', 'HUST_8-6.pkl', 'HUST_7-5.pkl']
HUST_vali =  ['HUST_6-8.pkl', 'HUST_3-8.pkl', 'HUST_2-3.pkl', 'HUST_9-1.pkl', 'HUST_10-8.pkl', 'HUST_7-4.pkl', 'HUST_2-8.pkl', 'HUST_8-3.pkl', 'HUST_5-2.pkl', 'HUST_10-1.pkl', 'HUST_5-5.pkl', 'HUST_5-7.pkl', 'HUST_7-8.pkl', 'HUST_7-6.pkl', 'HUST_1-8.pkl']
HUST_test = ['HUST_9-4.pkl', 'HUST_10-2.pkl', 'HUST_10-3.pkl', 'HUST_8-5.pkl', 'HUST_7-3.pkl', 'HUST_7-1.pkl', 'HUST_9-3.pkl', 'HUST_4-2.pkl', 'HUST_8-7.pkl', 'HUST_9-7.pkl', 'HUST_6-5.pkl', 'HUST_3-4.pkl', 'HUST_8-2.pkl', 'HUST_4-8.pkl', 'HUST_2-7.pkl']

ISU_ILCC_train = ['ISU-ILCC_G23C3.pkl', 'ISU-ILCC_G64C4.pkl', 'ISU-ILCC_G50C1.pkl', 'ISU-ILCC_G36C2.pkl', 'ISU-ILCC_G7C4.pkl', 'ISU-ILCC_G16C2.pkl', 'ISU-ILCC_G13C2.pkl', 'ISU-ILCC_G29C3.pkl', 'ISU-ILCC_G41C3.pkl', 'ISU-ILCC_G39C4.pkl', 'ISU-ILCC_G36C4.pkl', 'ISU-ILCC_G28C3.pkl', 'ISU-ILCC_G18C1.pkl', 'ISU-ILCC_G58C1.pkl', 'ISU-ILCC_G8C2.pkl', 'ISU-ILCC_G16C3.pkl', 'ISU-ILCC_G27C2.pkl', 'ISU-ILCC_G22C1.pkl', 'ISU-ILCC_G43C1.pkl', 'ISU-ILCC_G53C2.pkl', 'ISU-ILCC_G32C4.pkl', 'ISU-ILCC_G44C3.pkl', 'ISU-ILCC_G38C1.pkl', 'ISU-ILCC_G61C3.pkl', 'ISU-ILCC_G43C2.pkl', 'ISU-ILCC_G20C2.pkl', 'ISU-ILCC_G17C1.pkl', 'ISU-ILCC_G33C4.pkl', 'ISU-ILCC_G60C1.pkl', 'ISU-ILCC_G27C3.pkl', 'ISU-ILCC_G45C3.pkl', 'ISU-ILCC_G18C4.pkl', 'ISU-ILCC_G50C4.pkl', 'ISU-ILCC_G1C1.pkl', 'ISU-ILCC_G19C4.pkl', 'ISU-ILCC_G49C4.pkl', 'ISU-ILCC_G3C3.pkl', 'ISU-ILCC_G19C2.pkl', 'ISU-ILCC_G52C4.pkl', 'ISU-ILCC_G31C1.pkl', 'ISU-ILCC_G47C2.pkl', 'ISU-ILCC_G43C4.pkl', 'ISU-ILCC_G47C1.pkl', 'ISU-ILCC_G40C2.pkl', 'ISU-ILCC_G20C4.pkl', 'ISU-ILCC_G6C2.pkl', 'ISU-ILCC_G32C3.pkl', 'ISU-ILCC_G41C4.pkl', 'ISU-ILCC_G60C2.pkl', 'ISU-ILCC_G64C3.pkl', 'ISU-ILCC_G9C2.pkl', 'ISU-ILCC_G49C3.pkl', 'ISU-ILCC_G46C1.pkl', 'ISU-ILCC_G30C1.pkl', 'ISU-ILCC_G29C4.pkl', 'ISU-ILCC_G50C2.pkl', 'ISU-ILCC_G51C1.pkl', 'ISU-ILCC_G55C2.pkl', 'ISU-ILCC_G17C4.pkl', 'ISU-ILCC_G50C3.pkl', 'ISU-ILCC_G25C1.pkl', 'ISU-ILCC_G4C4.pkl', 'ISU-ILCC_G13C4.pkl', 'ISU-ILCC_G46C3.pkl', 'ISU-ILCC_G38C2.pkl', 'ISU-ILCC_G34C1.pkl', 'ISU-ILCC_G45C4.pkl', 'ISU-ILCC_G27C4.pkl', 'ISU-ILCC_G35C1.pkl', 'ISU-ILCC_G21C4.pkl', 'ISU-ILCC_G24C1.pkl', 'ISU-ILCC_G40C4.pkl', 'ISU-ILCC_G42C3.pkl', 'ISU-ILCC_G39C1.pkl', 'ISU-ILCC_G14C1.pkl', 'ISU-ILCC_G61C2.pkl', 'ISU-ILCC_G63C3.pkl', 'ISU-ILCC_G23C4.pkl', 'ISU-ILCC_G36C1.pkl', 'ISU-ILCC_G57C2.pkl', 'ISU-ILCC_G55C1.pkl', 'ISU-ILCC_G4C1.pkl', 'ISU-ILCC_G7C3.pkl', 'ISU-ILCC_G12C4.pkl', 'ISU-ILCC_G34C2.pkl', 'ISU-ILCC_G7C1.pkl', 'ISU-ILCC_G51C4.pkl', 'ISU-ILCC_G4C2.pkl', 'ISU-ILCC_G44C1.pkl', 'ISU-ILCC_G46C4.pkl', 'ISU-ILCC_G42C2.pkl', 'ISU-ILCC_G28C2.pkl', 'ISU-ILCC_G12C3.pkl', 'ISU-ILCC_G51C2.pkl', 'ISU-ILCC_G62C4.pkl', 'ISU-ILCC_G30C2.pkl', 'ISU-ILCC_G28C1.pkl', 'ISU-ILCC_G56C2.pkl', 'ISU-ILCC_G23C2.pkl', 'ISU-ILCC_G52C1.pkl', 'ISU-ILCC_G37C3.pkl', 'ISU-ILCC_G34C3.pkl', 'ISU-ILCC_G57C3.pkl', 'ISU-ILCC_G32C2.pkl', 'ISU-ILCC_G38C3.pkl', 'ISU-ILCC_G18C3.pkl', 'ISU-ILCC_G10C3.pkl', 'ISU-ILCC_G40C3.pkl', 'ISU-ILCC_G14C3.pkl', 'ISU-ILCC_G1C4.pkl', 'ISU-ILCC_G29C2.pkl', 'ISU-ILCC_G17C3.pkl', 'ISU-ILCC_G45C1.pkl', 'ISU-ILCC_G62C3.pkl', 'ISU-ILCC_G31C3.pkl', 'ISU-ILCC_G25C3.pkl', 'ISU-ILCC_G44C4.pkl', 'ISU-ILCC_G4C3.pkl', 'ISU-ILCC_G44C2.pkl', 'ISU-ILCC_G62C1.pkl', 'ISU-ILCC_G30C3.pkl', 'ISU-ILCC_G56C3.pkl', 'ISU-ILCC_G47C4.pkl', 'ISU-ILCC_G25C2.pkl', 'ISU-ILCC_G42C1.pkl', 'ISU-ILCC_G20C1.pkl', 'ISU-ILCC_G59C1.pkl', 'ISU-ILCC_G38C4.pkl', 'ISU-ILCC_G9C3.pkl', 'ISU-ILCC_G35C3.pkl', 'ISU-ILCC_G8C4.pkl', 'ISU-ILCC_G53C1.pkl', 'ISU-ILCC_G30C4.pkl', 'ISU-ILCC_G20C3.pkl', 'ISU-ILCC_G9C1.pkl', 'ISU-ILCC_G21C1.pkl', 'ISU-ILCC_G3C2.pkl', 'ISU-ILCC_G5C1.pkl', 'ISU-ILCC_G1C3.pkl', 'ISU-ILCC_G62C2.pkl', 'ISU-ILCC_G58C3.pkl', 'ISU-ILCC_G21C2.pkl', 'ISU-ILCC_G28C4.pkl', 'ISU-ILCC_G24C2.pkl']
ISU_ILCC_vali = ['ISU-ILCC_G57C1.pkl', 'ISU-ILCC_G31C4.pkl', 'ISU-ILCC_G6C1.pkl', 'ISU-ILCC_G39C3.pkl', 'ISU-ILCC_G7C2.pkl', 'ISU-ILCC_G16C4.pkl', 'ISU-ILCC_G10C4.pkl', 'ISU-ILCC_G10C2.pkl', 'ISU-ILCC_G64C1.pkl', 'ISU-ILCC_G35C2.pkl', 'ISU-ILCC_G12C1.pkl', 'ISU-ILCC_G49C2.pkl', 'ISU-ILCC_G54C2.pkl', 'ISU-ILCC_G19C3.pkl', 'ISU-ILCC_G33C1.pkl', 'ISU-ILCC_G63C1.pkl', 'ISU-ILCC_G34C4.pkl', 'ISU-ILCC_G54C1.pkl', 'ISU-ILCC_G6C4.pkl', 'ISU-ILCC_G45C2.pkl', 'ISU-ILCC_G13C3.pkl', 'ISU-ILCC_G59C3.pkl', 'ISU-ILCC_G48C2.pkl', 'ISU-ILCC_G48C1.pkl', 'ISU-ILCC_G60C4.pkl', 'ISU-ILCC_G55C4.pkl', 'ISU-ILCC_G46C2.pkl', 'ISU-ILCC_G49C1.pkl', 'ISU-ILCC_G10C1.pkl', 'ISU-ILCC_G31C2.pkl', 'ISU-ILCC_G57C4.pkl', 'ISU-ILCC_G22C4.pkl', 'ISU-ILCC_G64C2.pkl', 'ISU-ILCC_G32C1.pkl', 'ISU-ILCC_G52C2.pkl', 'ISU-ILCC_G17C2.pkl', 'ISU-ILCC_G18C2.pkl', 'ISU-ILCC_G59C2.pkl', 'ISU-ILCC_G2C1.pkl', 'ISU-ILCC_G47C3.pkl', 'ISU-ILCC_G48C4.pkl', 'ISU-ILCC_G27C1.pkl', 'ISU-ILCC_G19C1.pkl', 'ISU-ILCC_G36C3.pkl', 'ISU-ILCC_G21C3.pkl', 'ISU-ILCC_G53C4.pkl', 'ISU-ILCC_G63C2.pkl', 'ISU-ILCC_G29C1.pkl']
ISU_ILCC_test = ['ISU-ILCC_G24C4.pkl', 'ISU-ILCC_G54C3.pkl', 'ISU-ILCC_G14C4.pkl', 'ISU-ILCC_G35C4.pkl', 'ISU-ILCC_G24C3.pkl', 'ISU-ILCC_G41C2.pkl', 'ISU-ILCC_G56C4.pkl', 'ISU-ILCC_G5C3.pkl', 'ISU-ILCC_G41C1.pkl', 'ISU-ILCC_G6C3.pkl', 'ISU-ILCC_G2C4.pkl', 'ISU-ILCC_G53C3.pkl', 'ISU-ILCC_G59C4.pkl', 'ISU-ILCC_G48C3.pkl', 'ISU-ILCC_G40C1.pkl', 'ISU-ILCC_G8C3.pkl', 'ISU-ILCC_G43C3.pkl', 'ISU-ILCC_G39C2.pkl', 'ISU-ILCC_G2C2.pkl', 'ISU-ILCC_G2C3.pkl', 'ISU-ILCC_G51C3.pkl', 'ISU-ILCC_G63C4.pkl', 'ISU-ILCC_G33C3.pkl', 'ISU-ILCC_G12C2.pkl', 'ISU-ILCC_G22C2.pkl', 'ISU-ILCC_G52C3.pkl', 'ISU-ILCC_G5C4.pkl', 'ISU-ILCC_G3C1.pkl', 'ISU-ILCC_G22C3.pkl', 'ISU-ILCC_G55C3.pkl', 'ISU-ILCC_G54C4.pkl', 'ISU-ILCC_G5C2.pkl', 'ISU-ILCC_G61C4.pkl', 'ISU-ILCC_G16C1.pkl', 'ISU-ILCC_G37C1.pkl', 'ISU-ILCC_G58C4.pkl', 'ISU-ILCC_G23C1.pkl', 'ISU-ILCC_G37C4.pkl', 'ISU-ILCC_G58C2.pkl', 'ISU-ILCC_G8C1.pkl', 'ISU-ILCC_G14C2.pkl', 'ISU-ILCC_G37C2.pkl', 'ISU-ILCC_G56C1.pkl', 'ISU-ILCC_G33C2.pkl', 'ISU-ILCC_G3C4.pkl', 'ISU-ILCC_G60C3.pkl', 'ISU-ILCC_G13C1.pkl', 'ISU-ILCC_G1C2.pkl']

MATR_train = ['MATR_b1c5.pkl', 'MATR_b3c6.pkl', 'MATR_b1c7.pkl', 'MATR_b4c18.pkl', 'MATR_b3c44.pkl', 'MATR_b3c18.pkl', 'MATR_b2c12.pkl', 'MATR_b2c26.pkl', 'MATR_b4c26.pkl', 'MATR_b3c40.pkl', 'MATR_b1c9.pkl', 'MATR_b2c18.pkl', 'MATR_b3c17.pkl', 'MATR_b3c21.pkl', 'MATR_b2c6.pkl', 'MATR_b3c45.pkl', 'MATR_b1c23.pkl', 'MATR_b2c4.pkl', 'MATR_b1c35.pkl', 'MATR_b1c19.pkl', 'MATR_b2c39.pkl', 'MATR_b3c26.pkl', 'MATR_b2c28.pkl', 'MATR_b2c33.pkl', 'MATR_b3c39.pkl', 'MATR_b2c29.pkl', 'MATR_b3c31.pkl', 'MATR_b4c42.pkl', 'MATR_b4c7.pkl', 'MATR_b2c38.pkl', 'MATR_b1c41.pkl', 'MATR_b3c33.pkl', 'MATR_b4c40.pkl', 'MATR_b1c17.pkl', 'MATR_b3c1.pkl', 'MATR_b4c10.pkl', 'MATR_b1c33.pkl', 'MATR_b2c10.pkl', 'MATR_b3c13.pkl', 'MATR_b4c37.pkl', 'MATR_b4c23.pkl', 'MATR_b4c15.pkl', 'MATR_b2c0.pkl', 'MATR_b2c19.pkl', 'MATR_b4c1.pkl', 'MATR_b3c8.pkl', 'MATR_b1c15.pkl', 'MATR_b4c24.pkl', 'MATR_b3c15.pkl', 'MATR_b1c3.pkl', 'MATR_b1c16.pkl', 'MATR_b3c3.pkl', 'MATR_b4c20.pkl', 'MATR_b4c30.pkl', 'MATR_b4c25.pkl', 'MATR_b4c9.pkl', 'MATR_b1c20.pkl', 'MATR_b3c14.pkl', 'MATR_b2c5.pkl', 'MATR_b3c22.pkl', 'MATR_b3c16.pkl', 'MATR_b4c43.pkl', 'MATR_b4c19.pkl', 'MATR_b2c31.pkl', 'MATR_b2c21.pkl', 'MATR_b4c12.pkl', 'MATR_b2c36.pkl', 'MATR_b1c21.pkl', 'MATR_b2c3.pkl', 'MATR_b2c37.pkl', 'MATR_b4c4.pkl', 'MATR_b2c44.pkl', 'MATR_b4c34.pkl', 'MATR_b4c29.pkl', 'MATR_b3c7.pkl', 'MATR_b4c21.pkl', 'MATR_b2c1.pkl', 'MATR_b1c31.pkl', 'MATR_b2c14.pkl', 'MATR_b1c26.pkl', 'MATR_b4c38.pkl', 'MATR_b1c42.pkl', 'MATR_b2c17.pkl', 'MATR_b3c28.pkl', 'MATR_b3c10.pkl', 'MATR_b3c36.pkl', 'MATR_b3c24.pkl', 'MATR_b1c6.pkl', 'MATR_b2c34.pkl', 'MATR_b3c9.pkl', 'MATR_b4c14.pkl', 'MATR_b2c24.pkl', 'MATR_b2c30.pkl', 'MATR_b3c4.pkl', 'MATR_b4c11.pkl', 'MATR_b2c41.pkl', 'MATR_b4c8.pkl', 'MATR_b3c25.pkl', 'MATR_b1c38.pkl', 'MATR_b3c27.pkl', 'MATR_b1c18.pkl', 'MATR_b2c32.pkl']
MATR_vali =  ['MATR_b1c32.pkl', 'MATR_b1c36.pkl', 'MATR_b4c16.pkl', 'MATR_b1c25.pkl', 'MATR_b4c5.pkl', 'MATR_b3c11.pkl', 'MATR_b3c38.pkl', 'MATR_b4c39.pkl', 'MATR_b1c43.pkl', 'MATR_b1c11.pkl', 'MATR_b2c45.pkl', 'MATR_b4c3.pkl', 'MATR_b2c46.pkl', 'MATR_b2c40.pkl', 'MATR_b2c2.pkl', 'MATR_b2c47.pkl', 'MATR_b2c22.pkl', 'MATR_b3c35.pkl', 'MATR_b2c35.pkl', 'MATR_b1c2.pkl', 'MATR_b1c28.pkl', 'MATR_b4c28.pkl', 'MATR_b1c30.pkl', 'MATR_b2c25.pkl', 'MATR_b4c33.pkl', 'MATR_b4c31.pkl', 'MATR_b2c13.pkl', 'MATR_b1c27.pkl', 'MATR_b3c34.pkl', 'MATR_b4c32.pkl', 'MATR_b4c13.pkl', 'MATR_b1c14.pkl', 'MATR_b1c34.pkl', 'MATR_b4c22.pkl']
MATR_test = ['MATR_b2c20.pkl', 'MATR_b1c1.pkl', 'MATR_b1c0.pkl', 'MATR_b1c24.pkl', 'MATR_b1c39.pkl', 'MATR_b1c45.pkl', 'MATR_b3c30.pkl', 'MATR_b3c12.pkl', 'MATR_b1c37.pkl', 'MATR_b1c4.pkl', 'MATR_b1c44.pkl', 'MATR_b3c5.pkl', 'MATR_b3c19.pkl', 'MATR_b1c29.pkl', 'MATR_b4c36.pkl', 'MATR_b4c44.pkl', 'MATR_b4c2.pkl', 'MATR_b1c40.pkl', 'MATR_b3c41.pkl', 'MATR_b2c23.pkl', 'MATR_b3c29.pkl', 'MATR_b4c6.pkl', 'MATR_b3c20.pkl', 'MATR_b4c35.pkl', 'MATR_b3c0.pkl', 'MATR_b4c41.pkl', 'MATR_b4c27.pkl', 'MATR_b4c17.pkl', 'MATR_b2c27.pkl', 'MATR_b2c43.pkl', 'MATR_b2c11.pkl', 'MATR_b2c42.pkl', 'MATR_b4c0.pkl']

total_MICH_train = ['MICH_11C_pouch_NMC_-5C_0-100_0.2-1.5C.pkl', 'MICH_02C_pouch_NMC_-5C_0-100_0.2-0.2C.pkl', 'MICH_18H_pouch_NMC_45C_50-100_0.2-1.5C.pkl', 'MICH_04R_pouch_NMC_25C_0-100_1.5-1.5C.pkl', 'MICH_03H_pouch_NMC_45C_0-100_0.2-0.2C.pkl', 'MICH_08C_pouch_NMC_-5C_0-100_2-2C.pkl', 'MICH_17C_pouch_NMC_-5C_50-100_0.2-1.5C.pkl', 'MICH_10R_pouch_NMC_25C_0-100_0.2-1.5C.pkl', 'MICH_15H_pouch_NMC_45C_50-100_0.2-0.2C.pkl', 'MICH_13R_pouch_NMC_25C_50-100_0.2-0.2C.pkl', 'MICH_14C_pouch_NMC_-5C_50-100_0.2-0.2C.pkl', 'MICH_12H_pouch_NMC_45C_0-100_0.2-1.5C.pkl', 'MICH_BLForm2_pouch_NMC_45C_0-100_1-1C_b.pkl', 'MICH_BLForm1_pouch_NMC_45C_0-100_1-1C_a.pkl', 'MICH_MCForm32_pouch_NMC_45C_0-100_1-1C_b.pkl', 'MICH_MCForm27_pouch_NMC_25C_0-100_1-1C_g.pkl', 'MICH_MCForm22_pouch_NMC_25C_0-100_1-1C_b.pkl', 'MICH_MCForm36_pouch_NMC_45C_0-100_1-1C_f.pkl', 'MICH_MCForm29_pouch_NMC_25C_0-100_1-1C_i.pkl', 'MICH_MCForm39_pouch_NMC_45C_0-100_1-1C_i.pkl', 'MICH_BLForm5_pouch_NMC_45C_0-100_1-1C_e.pkl', 'MICH_BLForm16_pouch_NMC_25C_0-100_1-1C_f.pkl', 'MICH_MCForm30_pouch_NMC_25C_0-100_1-1C_j.pkl', 'MICH_BLForm14_pouch_NMC_25C_0-100_1-1C_d.pkl', 'MICH_MCForm38_pouch_NMC_45C_0-100_1-1C_h.pkl', 'MICH_BLForm8_pouch_NMC_45C_0-100_1-1C_h.pkl', 'MICH_BLForm17_pouch_NMC_25C_0-100_1-1C_g.pkl', 'MICH_MCForm23_pouch_NMC_25C_0-100_1-1C_c.pkl', 'MICH_BLForm18_pouch_NMC_25C_0-100_1-1C_h.pkl', 'MICH_BLForm10_pouch_NMC_25C_0-100_1-1C_j.pkl', 'MICH_MCForm35_pouch_NMC_45C_0-100_1-1C_e.pkl', 'MICH_MCForm25_pouch_NMC_25C_0-100_1-1C_e.pkl', 'MICH_MCForm26_pouch_NMC_25C_0-100_1-1C_f.pkl', 'MICH_MCForm37_pouch_NMC_45C_0-100_1-1C_g.pkl', 'MICH_BLForm15_pouch_NMC_25C_0-100_1-1C_e.pkl', 'MICH_MCForm31_pouch_NMC_45C_0-100_1-1C_a.pkl']
total_MICH_vali =  ['MICH_06H_pouch_NMC_45C_0-100_1.5-1.5C.pkl', 'MICH_07R_pouch_NMC_25C_0-100_2-2C.pkl', 'MICH_05C_pouch_NMC_-5C_0-100_1.5-1.5C.pkl', 'MICH_BLForm3_pouch_NMC_45C_0-100_1-1C_c.pkl', 'MICH_BLForm4_pouch_NMC_45C_0-100_1-1C_d.pkl', 'MICH_MCForm24_pouch_NMC_25C_0-100_1-1C_d.pkl', 'MICH_BLForm12_pouch_NMC_25C_0-100_1-1C_b.pkl', 'MICH_BLForm13_pouch_NMC_25C_0-100_1-1C_c.pkl', 'MICH_BLForm19_pouch_NMC_25C_0-100_1-1C_i.pkl', 'MICH_BLForm11_pouch_NMC_25C_0-100_1-1C_a.pkl', 'MICH_BLForm9_pouch_NMC_45C_0-100_1-1C_i.pkl']
total_MICH_test = ['MICH_09H_pouch_NMC_45C_0-100_2-2C.pkl', 'MICH_16R_pouch_NMC_25C_50-100_0.2-1.5C.pkl', 'MICH_01R_pouch_NMC_25C_0-100_0.2-0.2C.pkl', 'MICH_BLForm20_pouch_NMC_25C_0-100_1-1C_j.pkl', 'MICH_MCForm21_pouch_NMC_25C_0-100_1-1C_a.pkl', 'MICH_MCForm34_pouch_NMC_45C_0-100_1-1C_d.pkl', 'MICH_BLForm6_pouch_NMC_45C_0-100_1-1C_f.pkl', 'MICH_MCForm40_pouch_NMC_45C_0-100_1-1C_j.pkl', 'MICH_BLForm7_pouch_NMC_45C_0-100_1-1C_g.pkl', 'MICH_MCForm28_pouch_NMC_25C_0-100_1-1C_h.pkl', 'MICH_MCForm33_pouch_NMC_45C_0-100_1-1C_c.pkl']

RWTH_train = ['RWTH_016.pkl', 'RWTH_045.pkl', 'RWTH_009.pkl', 'RWTH_039.pkl', 'RWTH_046.pkl', 'RWTH_019.pkl', 'RWTH_037.pkl', 'RWTH_013.pkl', 'RWTH_003.pkl', 'RWTH_044.pkl', 'RWTH_026.pkl', 'RWTH_006.pkl', 'RWTH_031.pkl', 'RWTH_036.pkl', 'RWTH_048.pkl', 'RWTH_033.pkl', 'RWTH_021.pkl', 'RWTH_012.pkl', 'RWTH_034.pkl', 'RWTH_018.pkl', 'RWTH_022.pkl', 'RWTH_030.pkl', 'RWTH_028.pkl', 'RWTH_011.pkl', 'RWTH_040.pkl', 'RWTH_041.pkl', 'RWTH_042.pkl', 'RWTH_025.pkl', 'RWTH_047.pkl', 'RWTH_004.pkl']
RWTH_vali = ['RWTH_007.pkl', 'RWTH_032.pkl', 'RWTH_024.pkl', 'RWTH_002.pkl', 'RWTH_029.pkl', 'RWTH_010.pkl', 'RWTH_027.pkl', 'RWTH_014.pkl', 'RWTH_049.pkl']
RWTH_test = ['RWTH_038.pkl', 'RWTH_008.pkl', 'RWTH_035.pkl', 'RWTH_017.pkl', 'RWTH_015.pkl', 'RWTH_023.pkl', 'RWTH_020.pkl', 'RWTH_005.pkl', 'RWTH_043.pkl']

# BatteryML MIX100 does not have SNL batteries in the testing set. We resplit it.
SNL_train =  ['SNL_18650_NCA_25C_0-100_0.5-0.5C_b.pkl', 'SNL_18650_NCA_35C_0-100_0.5-1C_b.pkl', 'SNL_18650_NMC_35C_0-100_0.5-2C_b.pkl', 'SNL_18650_NMC_25C_0-100_0.5-3C_b.pkl', 'SNL_18650_NMC_25C_0-100_0.5-1C_a.pkl', 'SNL_18650_LFP_25C_0-100_0.5-3C_c.pkl', 'SNL_18650_NCA_25C_0-100_0.5-2C_b.pkl', 'SNL_18650_NMC_15C_0-100_0.5-2C_b.pkl', 'SNL_18650_LFP_35C_0-100_0.5-1C_d.pkl', 'SNL_18650_NCA_35C_0-100_0.5-1C_a.pkl', 'SNL_18650_LFP_35C_0-100_0.5-2C_b.pkl', 'SNL_18650_NMC_25C_0-100_0.5-1C_c.pkl', 'SNL_18650_NMC_25C_0-100_0.5-3C_a.pkl', 'SNL_18650_NCA_35C_0-100_0.5-1C_d.pkl', 'SNL_18650_NMC_25C_0-100_0.5-1C_d.pkl', 'SNL_18650_LFP_25C_0-100_0.5-3C_a.pkl', 'SNL_18650_NCA_15C_0-100_0.5-2C_a.pkl', 'SNL_18650_NMC_25C_0-100_0.5-2C_a.pkl', 'SNL_18650_NCA_35C_0-100_0.5-1C_c.pkl', 'SNL_18650_NCA_15C_0-100_0.5-1C_b.pkl', 'SNL_18650_NCA_35C_0-100_0.5-2C_a.pkl', 'SNL_18650_LFP_25C_0-100_0.5-3C_d.pkl', 'SNL_18650_NCA_15C_0-100_0.5-1C_a.pkl', 'SNL_18650_NMC_35C_0-100_0.5-2C_a.pkl', 'SNL_18650_NMC_15C_0-100_0.5-2C_a.pkl', 'SNL_18650_NCA_25C_20-80_0.5-0.5C_c.pkl', 'SNL_18650_NMC_25C_0-100_0.5-2C_b.pkl', 'SNL_18650_NMC_25C_0-100_0.5-0.5C_b.pkl', 'SNL_18650_NMC_35C_0-100_0.5-1C_a.pkl', 'SNL_18650_LFP_35C_0-100_0.5-1C_b.pkl']
SNL_vali = ['SNL_18650_NMC_25C_0-100_0.5-1C_b.pkl', 'SNL_18650_NMC_15C_0-100_0.5-1C_b.pkl', 'SNL_18650_NCA_25C_20-80_0.5-0.5C_d.pkl', 'SNL_18650_NCA_35C_0-100_0.5-2C_b.pkl', 'SNL_18650_NMC_25C_0-100_0.5-3C_d.pkl', 'SNL_18650_NCA_25C_0-100_0.5-1C_d.pkl', 'SNL_18650_NCA_25C_0-100_0.5-1C_a.pkl', 'SNL_18650_NCA_25C_0-100_0.5-1C_b.pkl', 'SNL_18650_NMC_35C_0-100_0.5-1C_d.pkl', 'SNL_18650_NCA_25C_0-100_0.5-0.5C_a.pkl']
SNL_test = ['SNL_18650_NMC_15C_0-100_0.5-1C_a.pkl', 'SNL_18650_NMC_35C_0-100_0.5-1C_c.pkl', 'SNL_18650_NCA_25C_0-100_0.5-2C_a.pkl', 'SNL_18650_NMC_25C_0-100_0.5-3C_c.pkl', 'SNL_18650_LFP_25C_0-100_0.5-3C_b.pkl', 'SNL_18650_NCA_15C_0-100_0.5-2C_b.pkl', 'SNL_18650_NMC_35C_0-100_0.5-1C_b.pkl', 'SNL_18650_LFP_35C_0-100_0.5-1C_c.pkl', 'SNL_18650_LFP_35C_0-100_0.5-2C_a.pkl', 'SNL_18650_NCA_25C_0-100_0.5-1C_c.pkl']

Stanford_train = ['Stanford_Nova_Regular_228.pkl', 'Stanford_Nova_Regular_203.pkl', 'Stanford_Nova_Regular_215.pkl', 'Stanford_Nova_Regular_225.pkl', 'Stanford_Nova_Regular_222.pkl', 'Stanford_Nova_Regular_226.pkl', 'Stanford_Nova_Regular_221.pkl', 'Stanford_Nova_Regular_211.pkl', 'Stanford_Nova_Regular_219.pkl', 'Stanford_Nova_Regular_229.pkl', 'Stanford_Nova_Regular_193.pkl', 'Stanford_Nova_Regular_200.pkl', 'Stanford_Nova_Regular_205.pkl', 'Stanford_Nova_Regular_230.pkl', 'Stanford_Nova_Regular_196.pkl', 'Stanford_Nova_Regular_216.pkl', 'Stanford_Nova_Regular_220.pkl', 'Stanford_Nova_Regular_201.pkl', 'Stanford_Nova_Regular_Ref_101.pkl', 'Stanford_Nova_Regular_192.pkl', 'Stanford_Nova_Regular_208.pkl', 'Stanford_Nova_Regular_Ref_102.pkl', 'Stanford_Nova_Regular_210.pkl', 'Stanford_Nova_Regular_212.pkl', 'Stanford_Nova_Regular_199.pkl']
Stanford_vali =['Stanford_Nova_Regular_217.pkl', 'Stanford_Nova_Regular_194.pkl', 'Stanford_Nova_Regular_214.pkl', 'Stanford_Nova_Regular_195.pkl', 'Stanford_Nova_Regular_191.pkl', 'Stanford_Nova_Regular_224.pkl', 'Stanford_Nova_Regular_206.pkl', 'Stanford_Nova_Regular_227.pkl']
Stanford_test = ['Stanford_Nova_Regular_202.pkl', 'Stanford_Nova_Regular_223.pkl', 'Stanford_Nova_Regular_209.pkl', 'Stanford_Nova_Regular_204.pkl', 'Stanford_Nova_Regular_213.pkl', 'Stanford_Nova_Regular_207.pkl', 'Stanford_Nova_Regular_198.pkl', 'Stanford_Nova_Regular_Ref_100.pkl']

Tongji_train =  ['Tongji1_CY35-05_1--2.pkl', 'Tongji1_CY45-05_1--21.pkl', 'Tongji1_CY45-05_1--28.pkl', 'Tongji2_CY45-05_1--11.pkl', 'Tongji1_CY25-1_1--5.pkl', 'Tongji1_CY25-1_1--9.pkl', 'Tongji1_CY45-05_1--19.pkl', 'Tongji3_CY25-05_4--1.pkl', 'Tongji1_CY25-05_1--2.pkl', 'Tongji3_CY25-05_2--1.pkl', 'Tongji1_CY45-05_1--5.pkl', 'Tongji1_CY45-05_1--13.pkl', 'Tongji1_CY45-05_1--11.pkl', 'Tongji1_CY35-05_1--1.pkl', 'Tongji1_CY45-05_1--17.pkl', 'Tongji2_CY25-05_1--13.pkl', 'Tongji1_CY25-1_1--6.pkl', 'Tongji2_CY45-05_1--22.pkl', 'Tongji1_CY45-05_1--27.pkl', 'Tongji2_CY45-05_1--27.pkl', 'Tongji1_CY25-05_1--17.pkl', 'Tongji2_CY45-05_1--9.pkl', 'Tongji3_CY25-05_2--2.pkl', 'Tongji1_CY45-05_1--18.pkl', 'Tongji3_CY25-05_4--2.pkl', 'Tongji1_CY45-05_1--20.pkl', 'Tongji1_CY45-05_1--1.pkl', 'Tongji2_CY45-05_1--23.pkl', 'Tongji2_CY25-05_1--5.pkl', 'Tongji2_CY45-05_1--25.pkl', 'Tongji1_CY25-05_1--4.pkl', 'Tongji1_CY25-05_1--12.pkl', 'Tongji1_CY25-05_1--16.pkl', 'Tongji1_CY25-05_1--13.pkl', 'Tongji2_CY25-05_1--17.pkl', 'Tongji1_CY45-05_1--7.pkl', 'Tongji1_CY25-025_1--5.pkl', 'Tongji2_CY25-05_1--12.pkl', 'Tongji1_CY45-05_1--6.pkl', 'Tongji3_CY25-05_1--2.pkl', 'Tongji3_CY25-05_2--3.pkl', 'Tongji2_CY45-05_1--16.pkl', 'Tongji1_CY25-05_1--15.pkl', 'Tongji1_CY25-025_1--3.pkl', 'Tongji2_CY45-05_1--7.pkl', 'Tongji2_CY45-05_1--20.pkl', 'Tongji1_CY45-05_1--23.pkl', 'Tongji2_CY25-05_1--15.pkl', 'Tongji2_CY35-05_1--4.pkl', 'Tongji1_CY45-05_1--12.pkl', 'Tongji2_CY45-05_1--15.pkl', 'Tongji1_CY25-05_1--5.pkl', 'Tongji2_CY35-05_1--3.pkl', 'Tongji1_CY25-05_1--14.pkl', 'Tongji1_CY45-05_1--26.pkl', 'Tongji1_CY45-05_1--22.pkl', 'Tongji2_CY45-05_1--24.pkl', 'Tongji1_CY25-05_1--18.pkl', 'Tongji1_CY45-05_1--9.pkl', 'Tongji1_CY45-05_1--24.pkl', 'Tongji1_CY25-1_1--4.pkl', 'Tongji2_CY45-05_1--26.pkl', 'Tongji3_CY25-05_1--3.pkl', 'Tongji1_CY45-05_1--15.pkl', 'Tongji1_CY25-025_1--2.pkl', 'Tongji2_CY45-05_1--14.pkl']
Tongji_vali = ['Tongji1_CY25-1_1--3.pkl', 'Tongji1_CY25-05_1--3.pkl', 'Tongji1_CY45-05_1--10.pkl', 'Tongji1_CY45-05_1--14.pkl', 'Tongji2_CY45-05_1--21.pkl', 'Tongji2_CY45-05_1--8.pkl', 'Tongji2_CY25-05_1--8.pkl', 'Tongji2_CY45-05_1--18.pkl', 'Tongji1_CY25-1_1--8.pkl', 'Tongji1_CY45-05_1--8.pkl', 'Tongji2_CY45-05_1--10.pkl', 'Tongji1_CY45-05_1--16.pkl', 'Tongji2_CY45-05_1--1.pkl', 'Tongji1_CY25-05_1--10.pkl', 'Tongji1_CY25-025_1--4.pkl', 'Tongji2_CY25-05_1--16.pkl', 'Tongji1_CY25-05_1--11.pkl', 'Tongji2_CY45-05_1--12.pkl', 'Tongji2_CY25-05_1--10.pkl', 'Tongji1_CY25-025_1--1.pkl', 'Tongji1_CY25-05_1--7.pkl']
Tongji_test =['Tongji1_CY45-05_1--2.pkl', 'Tongji1_CY45-05_1--25.pkl', 'Tongji2_CY25-05_1--9.pkl', 'Tongji1_CY25-1_1--1.pkl', 'Tongji1_CY25-05_1--19.pkl', 'Tongji1_CY25-1_1--2.pkl', 'Tongji2_CY45-05_1--17.pkl', 'Tongji1_CY25-025_1--7.pkl', 'Tongji2_CY45-05_1--19.pkl', 'Tongji2_CY45-05_1--13.pkl', 'Tongji1_CY25-025_1--6.pkl', 'Tongji2_CY35-05_1--2.pkl', 'Tongji2_CY45-05_1--2.pkl', 'Tongji1_CY25-05_1--1.pkl', 'Tongji3_CY25-05_4--3.pkl', 'Tongji3_CY25-05_1--1.pkl', 'Tongji1_CY25-1_1--7.pkl', 'Tongji2_CY45-05_1--28.pkl', 'Tongji2_CY35-05_1--1.pkl', 'Tongji2_CY25-05_1--2.pkl', 'Tongji1_CY25-05_1--6.pkl']

UL_PUR_train = ['UL-PUR_N10-NA7_18650_NCA_23C_0-100_0.5-0.5C_g.pkl', 'UL-PUR_N15-NA10_18650_NCA_23C_0-100_0.5-0.5C_j.pkl']
UL_PUR_vali = []
UL_PUR_test = []

XJTU_train = ['XJTU_2C_battery-6.pkl', 'XJTU_2C_battery-2.pkl', 'XJTU_3C_battery-2.pkl', 'XJTU_3C_battery-3.pkl', 'XJTU_3C_battery-10.pkl', 'XJTU_3C_battery-13.pkl', 'XJTU_3C_battery-7.pkl', 'XJTU_3C_battery-15.pkl', 'XJTU_3C_battery-4.pkl', 'XJTU_3C_battery-5.pkl', 'XJTU_3C_battery-11.pkl', 'XJTU_3C_battery-6.pkl', 'XJTU_2C_battery-1.pkl', 'XJTU_2C_battery-8.pkl', 'XJTU_2C_battery-4.pkl']
XJTU_vali = ['XJTU_3C_battery-14.pkl', 'XJTU_3C_battery-8.pkl', 'XJTU_2C_battery-3.pkl', 'XJTU_2C_battery-7.pkl']
XJTU_test = ['XJTU_3C_battery-12.pkl', 'XJTU_3C_battery-1.pkl', 'XJTU_2C_battery-5.pkl', 'XJTU_3C_battery-9.pkl']

ZNcoin_train_files = ['ZN-coin_202_20231213213655_03_3.pkl', 'ZN-coin_202_20231213213655_03_4.pkl',
                      'ZN-coin_202_20231213213655_03_5.pkl', 'ZN-coin_204-1_20231205230212_07_1.pkl',
                      'ZN-coin_204-3_20231205230221_07_3.pkl', 'ZN-coin_205-2_20231205230234_07_5.pkl',
                      'ZN-coin_402-2_20231209225727_01_2.pkl', 'ZN-coin_403-1_20231209225922_01_4.pkl',
                      'ZN-coin_404-3_20231209231250_08_1.pkl', 'ZN-coin_405-1_20231209231331_08_2.pkl',
                      'ZN-coin_405-2_20231209231413_08_3.pkl', 'ZN-coin_405-3_20231209231450_08_4.pkl',
                      'ZN-coin_407-1_20231209231725_08_8.pkl', 'ZN-coin_407-3_20231209231841_02_2.pkl',
                      'ZN-coin_408-1_20231209231918_02_3.pkl', 'ZN-coin_408-2_20231209231947_02_4.pkl',
                      'ZN-coin_408-3_20231209232028_05_1.pkl', 'ZN-coin_409-1_20231209232338_05_2.pkl',
                      'ZN-coin_409-2_20231209232422_05_3.pkl', 'ZN-coin_409-3_20231209232500_05_4.pkl',
                      'ZN-coin_410-1_20231209232559_09_1.pkl', 'ZN-coin_410-3_20231209232707_09_3.pkl',
                      'ZN-coin_412-3_20231209233120_06_1.pkl', 'ZN-coin_414-2_20231209233354_06_6.pkl',
                      'ZN-coin_415-2_20231209233606_10_1.pkl', 'ZN-coin_416-3_20231209233856_10_5.pkl',
                      'ZN-coin_417-3_20231209234058_10_8.pkl', 'ZN-coin_418-1_20231209234141_11_1.pkl',
                      'ZN-coin_418-3_20231209234252_11_3.pkl', 'ZN-coin_420-3_20231205230017_01_3.pkl',
                      'ZN-coin_422-3_20231205230049_02_1.pkl', 'ZN-coin_423-1_20231205230055_02_2.pkl',
                      'ZN-coin_425-2_20231205230124_03_1.pkl', 'ZN-coin_428-1_20231212185048_01_2.pkl',
                      'ZN-coin_429-2_20231212185157_01_8.pkl', 'ZN-coin_430-1_20231212185250_02_6.pkl',
                      'ZN-coin_432-2_20231227204437_01_2.pkl', 'ZN-coin_433-1_20231227204534_01_4.pkl',
                      'ZN-coin_433-2_20231227204539_01_5.pkl', 'ZN-coin_434-1_20231227204606_01_7.pkl',
                      'ZN-coin_434-2_20231227204612_01_8.pkl', 'ZN-coin_434-3_20231227204618_03_1.pkl',
                      'ZN-coin_435-2_20231227204630_03_3.pkl', 'ZN-coin_435-3_20231227204635_03_4.pkl',
                      'ZN-coin_436-3_20231227204657_03_7.pkl', 'ZN-coin_437-1_20231227204706_03_8.pkl',
                      'ZN-coin_437-3_20231227204717_04_2.pkl', 'ZN-coin_438-1_20231227204743_04_3.pkl',
                      'ZN-coin_438-2_20231227204748_04_4.pkl', 'ZN-coin_439-2_20231227204810_04_7.pkl',
                      'ZN-coin_439-3_20231227204817_04_8.pkl', 'ZN-coin_440-2_20231227204832_08_2.pkl',
                      'ZN-coin_442-1_20240104212418_09_1.pkl', 'ZN-coin_442-3_20240104212433_09_3.pkl',
                      'ZN-coin_443-2_20240104212500_09_5.pkl', 'ZN-coin_445-1_20240104212517_09_7.pkl',
                      'ZN-coin_450-1_20240116203402_01_2_Batch-3.pkl', 'ZN-coin_450-2_20240116203410_01_4_Batch-3.pkl',
                      'ZN-coin_450-3_20240116203417_03_3_Batch-3.pkl', 'ZN-coin_451-1_20240116203425_03_4_Batch-3.pkl']
ZNcoin_val_files = ['ZN-coin_442-2_20240104212424_09_2.pkl', 'ZN-coin_420-1_20231205230010_01_1.pkl',
                    'ZN-coin_209-2_20231205230252_07_8.pkl', 'ZN-coin_441-1_20231227204855_08_4.pkl',
                    'ZN-coin_437-2_20231227204712_04_1.pkl', 'ZN-coin_406-1_20231209231531_08_5.pkl',
                    'ZN-coin_411-1_20231209232756_09_4.pkl', 'ZN-coin_436-1_20231227204646_03_5.pkl',
                    'ZN-coin_418-2_20231209234209_11_2.pkl', 'ZN-coin_440-3_20231227204837_08_3.pkl',
                    'ZN-coin_429-1_20231212185129_01_5.pkl', 'ZN-coin_432-3_20231227204518_01_3.pkl',
                    'ZN-coin_433-3_20231227204544_01_6.pkl', 'ZN-coin_445-2_20240104212521_09_8.pkl',
                    'ZN-coin_414-3_20231209233430_06_7.pkl', 'ZN-coin_416-2_20231209233822_10_4.pkl',
                    'ZN-coin_445-3_20240104212530_07_1.pkl', 'ZN-coin_440-1_20231227204827_08_1.pkl',
                    'ZN-coin_415-3_20231209233637_10_2.pkl', 'ZN-coin_402-1_20231209225636_01_1.pkl']
ZNcoin_test_files = ['ZN-coin_422-1_20231205230039_01_7.pkl', 'ZN-coin_438-3_20231227204754_04_5.pkl',
                     'ZN-coin_435-1_20231227204625_03_2.pkl', 'ZN-coin_412-2_20231209233028_09_8.pkl',
                     'ZN-coin_410-2_20231209232626_09_2.pkl', 'ZN-coin_439-1_20231227204804_04_6.pkl',
                     'ZN-coin_204-2_20231205230217_07_2.pkl', 'ZN-coin_428-2_20231212185058_01_4.pkl',
                     'ZN-coin_430-2_20231212185305_02_7.pkl', 'ZN-coin_436-2_20231227204653_03_6.pkl',
                     'ZN-coin_205-3_20231205230239_07_6.pkl', 'ZN-coin_415-1_20231209233508_06_8.pkl',
                     'ZN-coin_412-1_20231209232958_09_7.pkl', 'ZN-coin_413-1_20231209233202_06_2.pkl',
                     'ZN-coin_209-1_20231205230248_07_7.pkl', 'ZN-coin_406-2_20231209231604_08_6.pkl',
                     'ZN-coin_406-3_20231209231637_08_7.pkl', 'ZN-coin_205-1_20231205230230_07_4.pkl',
                     'ZN-coin_446-1_20240104212538_07_2.pkl', 'ZN-coin_402-3_20231209225844_01_3.pkl']

ZN_2024_train_files = ['ZN-coin_202_20231213213655_03_3.pkl', 'ZN-coin_202_20231213213655_03_4.pkl', 'ZN-coin_202_20231213213655_03_5.pkl', 'ZN-coin_204-1_20231205230212_07_1.pkl', 'ZN-coin_204-2_20231205230217_07_2.pkl', 'ZN-coin_204-3_20231205230221_07_3.pkl', 'ZN-coin_205-1_20231205230230_07_4.pkl', 'ZN-coin_205-3_20231205230239_07_6.pkl', 'ZN-coin_209-2_20231205230252_07_8.pkl', 'ZN-coin_402-1_20231209225636_01_1.pkl', 'ZN-coin_402-2_20231209225727_01_2.pkl', 'ZN-coin_402-3_20231209225844_01_3.pkl', 'ZN-coin_403-1_20231209225922_01_4.pkl', 'ZN-coin_404-3_20231209231250_08_1.pkl', 'ZN-coin_405-1_20231209231331_08_2.pkl', 'ZN-coin_405-2_20231209231413_08_3.pkl', 'ZN-coin_406-1_20231209231531_08_5.pkl', 'ZN-coin_406-2_20231209231604_08_6.pkl', 'ZN-coin_406-3_20231209231637_08_7.pkl', 'ZN-coin_407-1_20231209231725_08_8.pkl', 'ZN-coin_408-1_20231209231918_02_3.pkl', 'ZN-coin_408-3_20231209232028_05_1.pkl', 'ZN-coin_409-1_20231209232338_05_2.pkl', 'ZN-coin_410-1_20231209232559_09_1.pkl', 'ZN-coin_410-3_20231209232707_09_3.pkl', 'ZN-coin_412-1_20231209232958_09_7.pkl', 'ZN-coin_412-2_20231209233028_09_8.pkl', 'ZN-coin_412-3_20231209233120_06_1.pkl', 'ZN-coin_413-1_20231209233202_06_2.pkl', 'ZN-coin_414-3_20231209233430_06_7.pkl', 'ZN-coin_415-1_20231209233508_06_8.pkl', 'ZN-coin_415-2_20231209233606_10_1.pkl', 'ZN-coin_415-3_20231209233637_10_2.pkl', 'ZN-coin_416-2_20231209233822_10_4.pkl', 'ZN-coin_418-1_20231209234141_11_1.pkl', 'ZN-coin_418-2_20231209234209_11_2.pkl', 'ZN-coin_420-1_20231205230010_01_1.pkl', 'ZN-coin_422-1_20231205230039_01_7.pkl', 'ZN-coin_428-2_20231212185058_01_4.pkl', 'ZN-coin_429-1_20231212185129_01_5.pkl', 'ZN-coin_429-2_20231212185157_01_8.pkl', 'ZN-coin_430-1_20231212185250_02_6.pkl', 'ZN-coin_433-2_20231227204539_01_5.pkl', 'ZN-coin_433-3_20231227204544_01_6.pkl', 'ZN-coin_434-1_20231227204606_01_7.pkl', 'ZN-coin_435-1_20231227204625_03_2.pkl', 'ZN-coin_435-2_20231227204630_03_3.pkl', 'ZN-coin_436-1_20231227204646_03_5.pkl', 'ZN-coin_436-2_20231227204653_03_6.pkl', 'ZN-coin_437-1_20231227204706_03_8.pkl', 'ZN-coin_437-2_20231227204712_04_1.pkl', 'ZN-coin_437-3_20231227204717_04_2.pkl', 'ZN-coin_438-1_20231227204743_04_3.pkl', 'ZN-coin_438-3_20231227204754_04_5.pkl', 'ZN-coin_440-1_20231227204827_08_1.pkl', 'ZN-coin_440-3_20231227204837_08_3.pkl', 'ZN-coin_441-1_20231227204855_08_4.pkl', 'ZN-coin_442-2_20240104212424_09_2.pkl', 'ZN-coin_442-3_20240104212433_09_3.pkl', 'ZN-coin_446-1_20240104212538_07_2.pkl']
ZN_2024_val_files = ['ZN-coin_450-3_20240116203417_03_3_Batch-3.pkl', 'ZN-coin_409-3_20231209232500_05_4.pkl', 'ZN-coin_416-3_20231209233856_10_5.pkl', 'ZN-coin_439-3_20231227204817_04_8.pkl', 'ZN-coin_418-3_20231209234252_11_3.pkl', 'ZN-coin_438-2_20231227204748_04_4.pkl', 'ZN-coin_209-1_20231205230248_07_7.pkl', 'ZN-coin_409-2_20231209232422_05_3.pkl', 'ZN-coin_435-3_20231227204635_03_4.pkl', 'ZN-coin_405-3_20231209231450_08_4.pkl', 'ZN-coin_439-2_20231227204810_04_7.pkl', 'ZN-coin_451-1_20240116203425_03_4_Batch-3.pkl', 'ZN-coin_432-2_20231227204437_01_2.pkl', 'ZN-coin_205-2_20231205230234_07_5.pkl', 'ZN-coin_420-3_20231205230017_01_3.pkl', 'ZN-coin_432-3_20231227204518_01_3.pkl', 'ZN-coin_440-2_20231227204832_08_2.pkl', 'ZN-coin_442-1_20240104212418_09_1.pkl', 'ZN-coin_425-2_20231205230124_03_1.pkl', 'ZN-coin_428-1_20231212185048_01_2.pkl']
ZN_2024_test_files = ['ZN-coin_430-2_20231212185305_02_7.pkl', 'ZN-coin_407-3_20231209231841_02_2.pkl', 'ZN-coin_445-2_20240104212521_09_8.pkl', 'ZN-coin_436-3_20231227204657_03_7.pkl', 'ZN-coin_414-2_20231209233354_06_6.pkl', 'ZN-coin_408-2_20231209231947_02_4.pkl', 'ZN-coin_445-1_20240104212517_09_7.pkl', 'ZN-coin_422-3_20231205230049_02_1.pkl', 'ZN-coin_450-1_20240116203402_01_2_Batch-3.pkl', 'ZN-coin_443-2_20240104212500_09_5.pkl', 'ZN-coin_450-2_20240116203410_01_4_Batch-3.pkl', 'ZN-coin_411-1_20231209232756_09_4.pkl', 'ZN-coin_434-3_20231227204618_03_1.pkl', 'ZN-coin_410-2_20231209232626_09_2.pkl', 'ZN-coin_439-1_20231227204804_04_6.pkl', 'ZN-coin_445-3_20240104212530_07_1.pkl', 'ZN-coin_433-1_20231227204534_01_4.pkl', 'ZN-coin_417-3_20231209234058_10_8.pkl', 'ZN-coin_423-1_20231205230055_02_2.pkl', 'ZN-coin_434-2_20231227204612_01_8.pkl']

ZN_42_train_files = ['ZN-coin_202_20231213213655_03_4.pkl', 'ZN-coin_202_20231213213655_03_5.pkl', 'ZN-coin_204-3_20231205230221_07_3.pkl', 'ZN-coin_205-1_20231205230230_07_4.pkl', 'ZN-coin_205-2_20231205230234_07_5.pkl', 'ZN-coin_209-1_20231205230248_07_7.pkl', 'ZN-coin_209-2_20231205230252_07_8.pkl', 'ZN-coin_402-2_20231209225727_01_2.pkl', 'ZN-coin_404-3_20231209231250_08_1.pkl', 'ZN-coin_405-3_20231209231450_08_4.pkl', 'ZN-coin_406-2_20231209231604_08_6.pkl', 'ZN-coin_406-3_20231209231637_08_7.pkl', 'ZN-coin_407-1_20231209231725_08_8.pkl', 'ZN-coin_407-3_20231209231841_02_2.pkl', 'ZN-coin_408-1_20231209231918_02_3.pkl', 'ZN-coin_410-1_20231209232559_09_1.pkl', 'ZN-coin_411-1_20231209232756_09_4.pkl', 'ZN-coin_412-1_20231209232958_09_7.pkl', 'ZN-coin_413-1_20231209233202_06_2.pkl', 'ZN-coin_415-1_20231209233508_06_8.pkl', 'ZN-coin_415-2_20231209233606_10_1.pkl', 'ZN-coin_415-3_20231209233637_10_2.pkl', 'ZN-coin_416-2_20231209233822_10_4.pkl', 'ZN-coin_416-3_20231209233856_10_5.pkl', 'ZN-coin_417-3_20231209234058_10_8.pkl', 'ZN-coin_418-2_20231209234209_11_2.pkl', 'ZN-coin_418-3_20231209234252_11_3.pkl', 'ZN-coin_420-1_20231205230010_01_1.pkl', 'ZN-coin_420-3_20231205230017_01_3.pkl', 'ZN-coin_422-1_20231205230039_01_7.pkl', 'ZN-coin_422-3_20231205230049_02_1.pkl', 'ZN-coin_423-1_20231205230055_02_2.pkl', 'ZN-coin_428-2_20231212185058_01_4.pkl', 'ZN-coin_429-1_20231212185129_01_5.pkl', 'ZN-coin_429-2_20231212185157_01_8.pkl', 'ZN-coin_430-1_20231212185250_02_6.pkl', 'ZN-coin_432-2_20231227204437_01_2.pkl', 'ZN-coin_432-3_20231227204518_01_3.pkl', 'ZN-coin_433-1_20231227204534_01_4.pkl', 'ZN-coin_433-3_20231227204544_01_6.pkl', 'ZN-coin_434-3_20231227204618_03_1.pkl', 'ZN-coin_435-2_20231227204630_03_3.pkl', 'ZN-coin_436-2_20231227204653_03_6.pkl', 'ZN-coin_436-3_20231227204657_03_7.pkl', 'ZN-coin_437-2_20231227204712_04_1.pkl', 'ZN-coin_438-1_20231227204743_04_3.pkl', 'ZN-coin_438-2_20231227204748_04_4.pkl', 'ZN-coin_438-3_20231227204754_04_5.pkl', 'ZN-coin_439-2_20231227204810_04_7.pkl', 'ZN-coin_439-3_20231227204817_04_8.pkl', 'ZN-coin_440-1_20231227204827_08_1.pkl', 'ZN-coin_440-2_20231227204832_08_2.pkl', 'ZN-coin_441-1_20231227204855_08_4.pkl', 'ZN-coin_443-2_20240104212500_09_5.pkl', 'ZN-coin_445-1_20240104212517_09_7.pkl', 'ZN-coin_446-1_20240104212538_07_2.pkl', 'ZN-coin_450-1_20240116203402_01_2_Batch-3.pkl', 'ZN-coin_450-2_20240116203410_01_4_Batch-3.pkl', 'ZN-coin_450-3_20240116203417_03_3_Batch-3.pkl', 'ZN-coin_451-1_20240116203425_03_4_Batch-3.pkl']
ZN_42_val_files = ['ZN-coin_412-3_20231209233120_06_1.pkl', 'ZN-coin_442-1_20240104212418_09_1.pkl', 'ZN-coin_434-1_20231227204606_01_7.pkl', 'ZN-coin_414-3_20231209233430_06_7.pkl', 'ZN-coin_436-1_20231227204646_03_5.pkl', 'ZN-coin_418-1_20231209234141_11_1.pkl', 'ZN-coin_202_20231213213655_03_3.pkl', 'ZN-coin_408-3_20231209232028_05_1.pkl', 'ZN-coin_434-2_20231227204612_01_8.pkl', 'ZN-coin_428-1_20231212185048_01_2.pkl', 'ZN-coin_445-2_20240104212521_09_8.pkl', 'ZN-coin_408-2_20231209231947_02_4.pkl', 'ZN-coin_414-2_20231209233354_06_6.pkl', 'ZN-coin_442-2_20240104212424_09_2.pkl', 'ZN-coin_406-1_20231209231531_08_5.pkl', 'ZN-coin_405-1_20231209231331_08_2.pkl', 'ZN-coin_430-2_20231212185305_02_7.pkl', 'ZN-coin_205-3_20231205230239_07_6.pkl', 'ZN-coin_410-3_20231209232707_09_3.pkl', 'ZN-coin_442-3_20240104212433_09_3.pkl']
ZN_42_test_files = ['ZN-coin_439-1_20231227204804_04_6.pkl', 'ZN-coin_403-1_20231209225922_01_4.pkl', 'ZN-coin_204-1_20231205230212_07_1.pkl', 'ZN-coin_445-3_20240104212530_07_1.pkl', 'ZN-coin_412-2_20231209233028_09_8.pkl', 'ZN-coin_410-2_20231209232626_09_2.pkl', 'ZN-coin_409-2_20231209232422_05_3.pkl', 'ZN-coin_405-2_20231209231413_08_3.pkl', 'ZN-coin_402-3_20231209225844_01_3.pkl', 'ZN-coin_440-3_20231227204837_08_3.pkl', 'ZN-coin_435-1_20231227204625_03_2.pkl', 'ZN-coin_402-1_20231209225636_01_1.pkl', 'ZN-coin_437-1_20231227204706_03_8.pkl', 'ZN-coin_425-2_20231205230124_03_1.pkl', 'ZN-coin_204-2_20231205230217_07_2.pkl', 'ZN-coin_409-1_20231209232338_05_2.pkl', 'ZN-coin_409-3_20231209232500_05_4.pkl', 'ZN-coin_433-2_20231227204539_01_5.pkl', 'ZN-coin_437-3_20231227204717_04_2.pkl', 'ZN-coin_435-3_20231227204635_03_4.pkl']

CALB_422_train_files = ['CALB_0_B183.pkl', 'CALB_0_B184.pkl', 'CALB_0_B187.pkl', 'CALB_0_B190.pkl', 'CALB_35_B250.pkl',
                        'CALB_35_B174.pkl', 'CALB_35_B175.pkl', 'CALB_35_B222.pkl', 'CALB_35_B223.pkl',
                        'CALB_35_B224.pkl', 'CALB_35_B227.pkl', 'CALB_35_B228.pkl', 'CALB_35_B229.pkl',
                        'CALB_35_B230.pkl', 'CALB_35_B249.pkl', 'CALB_45_B253.pkl', 'CALB_45_B255.pkl']
CALB_422_val_files = ['CALB_35_B173.pkl', 'CALB_0_B189.pkl', 'CALB_0_B188.pkl', 'CALB_45_B256.pkl', 'CALB_35_B248.pkl']
CALB_422_test_files = ['CALB_35_B247.pkl', 'CALB_0_B185.pkl', 'CALB_0_B182.pkl', 'CALB_25_T25-2.pkl',
                       'CALB_25_T25-1.pkl']

CALB_2024_train_files = ['CALB_0_B182.pkl', 'CALB_0_B183.pkl', 'CALB_0_B184.pkl', 'CALB_0_B185.pkl', 'CALB_0_B187.pkl',
                         'CALB_0_B189.pkl', 'CALB_35_B174.pkl', 'CALB_35_B175.pkl', 'CALB_35_B222.pkl',
                         'CALB_35_B223.pkl', 'CALB_35_B228.pkl', 'CALB_35_B230.pkl', 'CALB_35_B247.pkl',
                         'CALB_35_B249.pkl', 'CALB_45_B253.pkl', 'CALB_45_B255.pkl', 'CALB_45_B256.pkl']
CALB_2024_val_files = ['CALB_0_B190.pkl', 'CALB_35_B227.pkl', 'CALB_35_B173.pkl', 'CALB_35_B248.pkl',
                       'CALB_35_B229.pkl']
CALB_2024_test_files = ['CALB_35_B224.pkl', 'CALB_0_B188.pkl', 'CALB_35_B250.pkl', 'CALB_25_T25-1.pkl',
                        'CALB_25_T25-2.pkl']

CALB_train_files = ['CALB_35_B229.pkl', 'CALB_35_B173.pkl', 'CALB_35_B228.pkl', 'CALB_0_B184.pkl', 'CALB_35_B248.pkl',
                    'CALB_35_B227.pkl', 'CALB_0_B185.pkl', 'CALB_35_B249.pkl', 'CALB_35_B223.pkl', 'CALB_35_B224.pkl',
                    'CALB_0_B189.pkl', 'CALB_35_B250.pkl', 'CALB_0_B188.pkl', 'CALB_45_B256.pkl', 'CALB_0_B183.pkl',
                    'CALB_35_B175.pkl', 'CALB_0_B190.pkl']
CALB_val_files = ['CALB_0_B187.pkl', 'CALB_35_B222.pkl', 'CALB_25_T25-2.pkl', 'CALB_35_B247.pkl', 'CALB_45_B253.pkl']
CALB_test_files = ['CALB_0_B182.pkl', 'CALB_25_T25-1.pkl', 'CALB_35_B174.pkl', 'CALB_35_B230.pkl', 'CALB_45_B255.pkl']

NAion_42_train_files = ['NA-ion_270040-1-2-63.pkl', 'NA-ion_270040-1-5-60.pkl', 'NA-ion_270040-2-2-12.pkl',
                        'NA-ion_270040-3-1-56.pkl', 'NA-ion_270040-3-2-55.pkl', 'NA-ion_270040-3-3-54.pkl',
                        'NA-ion_270040-3-4-53.pkl', 'NA-ion_270040-3-5-52.pkl', 'NA-ion_270040-4-3-46.pkl',
                        'NA-ion_270040-3-8-49.pkl', 'NA-ion_270040-4-1-48.pkl', 'NA-ion_270040-4-2-47.pkl',
                        'NA-ion_270040-4-6-43.pkl', 'NA-ion_270040-5-1-39.pkl', 'NA-ion_270040-5-2-38.pkl',
                        'NA-ion_270040-5-6-34.pkl', 'NA-ion_270040-5-7-33.pkl', 'NA-ion_270040-6-2-30.pkl',
                        'NA-ion_270040-6-8-24.pkl', 'NA-ion_270040-7-1-23.pkl']
NAion_42_val_files = ['NA-ion_270040-6-6-26.pkl', 'NA-ion_270040-1-8-57.pkl', 'NA-ion_270040-5-8-32.pkl',
                      'NA-ion_270040-5-3-37.pkl', 'NA-ion_270040-1-6-59.pkl', 'NA-ion_270040-5-5-35.pkl']
NAion_42_test_files = ['NA-ion_270040-2-5-12.pkl', 'NA-ion_270040-1-3-62.pkl', 'NA-ion_270040-3-7-50.pkl',
                       'NA-ion_270040-8-5-16.pkl', 'NA-ion_270040-1-7-58.pkl']

NAion_2021_train_files = ['NA-ion_270040-1-2-63.pkl', 'NA-ion_270040-1-5-60.pkl', 'NA-ion_270040-1-7-58.pkl',
                          'NA-ion_270040-1-8-57.pkl', 'NA-ion_270040-2-2-12.pkl', 'NA-ion_270040-2-5-12.pkl',
                          'NA-ion_270040-3-1-56.pkl', 'NA-ion_270040-3-5-52.pkl', 'NA-ion_270040-5-2-38.pkl',
                          'NA-ion_270040-3-8-49.pkl', 'NA-ion_270040-5-1-39.pkl', 'NA-ion_270040-5-3-37.pkl',
                          'NA-ion_270040-5-6-34.pkl', 'NA-ion_270040-5-7-33.pkl', 'NA-ion_270040-6-2-30.pkl',
                          'NA-ion_270040-6-6-26.pkl', 'NA-ion_270040-7-1-23.pkl', 'NA-ion_270040-8-5-16.pkl',
                          'NA-ion_270040-3-3-54.pkl', 'NA-ion_270040-6-8-24.pkl']
NAion_2021_val_files = ['NA-ion_270040-4-2-47.pkl', 'NA-ion_270040-4-6-43.pkl', 'NA-ion_270040-5-5-35.pkl',
                        'NA-ion_270040-1-6-59.pkl', 'NA-ion_270040-3-4-53.pkl', 'NA-ion_270040-3-2-55.pkl']
NAion_2021_test_files = ['NA-ion_270040-5-8-32.pkl', 'NA-ion_270040-4-3-46.pkl', 'NA-ion_270040-4-1-48.pkl',
                         'NA-ion_270040-1-3-62.pkl', 'NA-ion_270040-3-7-50.pkl']

NAion_2024_train_files = ['NA-ion_270040-1-2-63.pkl', 'NA-ion_270040-6-8-24.pkl', 'NA-ion_270040-1-5-60.pkl',
                          'NA-ion_270040-1-6-59.pkl', 'NA-ion_270040-1-7-58.pkl', 'NA-ion_270040-1-8-57.pkl',
                          'NA-ion_270040-2-5-12.pkl', 'NA-ion_270040-3-3-54.pkl', 'NA-ion_270040-3-5-52.pkl',
                          'NA-ion_270040-2-2-12.pkl', 'NA-ion_270040-3-8-49.pkl', 'NA-ion_270040-4-2-47.pkl',
                          'NA-ion_270040-4-3-46.pkl', 'NA-ion_270040-4-6-43.pkl', 'NA-ion_270040-5-3-37.pkl',
                          'NA-ion_270040-5-5-35.pkl', 'NA-ion_270040-5-7-33.pkl', 'NA-ion_270040-5-8-32.pkl',
                          'NA-ion_270040-6-6-26.pkl', 'NA-ion_270040-5-6-34.pkl']
NAion_2024_val_files = ['NA-ion_270040-4-1-48.pkl', 'NA-ion_270040-7-1-23.pkl', 'NA-ion_270040-6-2-30.pkl',
                        'NA-ion_270040-3-2-55.pkl', 'NA-ion_270040-5-2-38.pkl', 'NA-ion_270040-3-1-56.pkl']
NAion_2024_test_files = ['NA-ion_270040-8-5-16.pkl', 'NA-ion_270040-3-4-53.pkl', 'NA-ion_270040-5-1-39.pkl',
                         'NA-ion_270040-3-7-50.pkl', 'NA-ion_270040-1-3-62.pkl']


def find_dataset(dataset_name, random_seed, type):
    dataset_name = str(dataset_name)
    random_seed = int(random_seed)

    if dataset_name == 'CALCE' and type == 'train':
        return CALCE_train
    elif dataset_name == 'CALCE' and type == 'vali':
        return CALCE_vali
    elif dataset_name == 'CALCE' and type == 'test':
        return CALCE_test
    elif dataset_name == 'HNEI' and type == 'train':
        return HNEI_train
    elif dataset_name == 'HNEI' and type == 'vali':
        return HNEI_vali
    elif dataset_name == 'HNEI' and type == 'test':
        return HNEI_test
    elif dataset_name == 'HUST' and type == 'train':
        return HUST_train
    elif dataset_name == 'HUST' and type == 'vali':
        return HUST_vali
    elif dataset_name == 'HUST' and type == 'test':
        return HUST_test
    elif dataset_name == 'ISU_ILCC' and type == 'train':
        return ISU_ILCC_train
    elif dataset_name == 'ISU_ILCC' and type == 'vali':
        return ISU_ILCC_vali
    elif dataset_name == 'ISU_ILCC' and type == 'test':
        return ISU_ILCC_test
    elif dataset_name == 'MATR' and type == 'train':
        return MATR_train
    elif dataset_name == 'MATR' and type == 'vali':
        return MATR_vali
    elif dataset_name == 'MATR' and type == 'test':
        return MATR_test
    elif dataset_name == 'total_MICH' and type == 'train':
        return total_MICH_train
    elif dataset_name == 'total_MICH' and type == 'vali':
        return total_MICH_vali
    elif dataset_name == 'total_MICH' and type == 'test':
        return total_MICH_test
    elif dataset_name == 'RWTH' and type == 'train':
        return RWTH_train
    elif dataset_name == 'RWTH' and type == 'vali':
        return RWTH_vali
    elif dataset_name == 'RWTH' and type == 'test':
        return RWTH_test
    elif dataset_name == 'SNL' and type == 'train':
        return SNL_train
    elif dataset_name == 'SNL' and type == 'vali':
        return SNL_vali
    elif dataset_name == 'SNL' and type == 'test':
        return SNL_test
    elif dataset_name == 'Stanford' and type == 'train':
        return Stanford_train
    elif dataset_name == 'Stanford' and type == 'vali':
        return Stanford_vali
    elif dataset_name == 'Stanford' and type == 'test':
        return Stanford_test
    elif dataset_name == 'Tongji' and type == 'train':
        return Tongji_train
    elif dataset_name == 'Tongji' and type == 'vali':
        return Tongji_vali
    elif dataset_name == 'Tongji' and type == 'test':
        return Tongji_test
    elif dataset_name == 'XJTU' and type == 'train':
        return XJTU_train
    elif dataset_name == 'XJTU' and type == 'vali':
        return XJTU_vali
    elif dataset_name == 'XJTU' and type == 'test':
        return XJTU_test
    elif dataset_name == 'UL_PUR' and type == 'train':
        return UL_PUR_train
    elif dataset_name == 'UL_PUR' and type == 'vali':
        return UL_PUR_vali
    elif dataset_name == 'UL_PUR' and type == 'test':
        return UL_PUR_test
    elif dataset_name == 'ZN-coin' and type == 'train' and random_seed == 2021:
        return ZNcoin_train_files
    elif dataset_name == 'ZN-coin' and type == 'vali' and random_seed == 2021:
        return ZNcoin_val_files
    elif dataset_name == 'ZN-coin' and type == 'test' and random_seed == 2021:
        return ZNcoin_test_files
    elif dataset_name == 'ZN-coin' and type == 'train' and random_seed == 2024:
        return ZN_2024_train_files
    elif dataset_name == 'ZN-coin' and type == 'vali' and random_seed == 2024:
        return ZN_2024_val_files
    elif dataset_name == 'ZN-coin' and type == 'test' and random_seed == 2024:
        return ZN_2024_test_files
    elif dataset_name == 'ZN-coin' and type == 'train' and random_seed == 42:
        return ZN_42_train_files
    elif dataset_name == 'ZN-coin' and type == 'vali' and random_seed == 42:
        return ZN_42_val_files
    elif dataset_name == 'ZN-coin' and type == 'test' and random_seed == 42:
        return ZN_42_test_files
    elif dataset_name == 'CALB' and type == 'train' and random_seed == 2021:
        return CALB_train_files
    elif dataset_name == 'CALB' and type == 'vali' and random_seed == 2021:
        return CALB_val_files
    elif dataset_name == 'CALB' and type == 'test' and random_seed == 2021:
        return CALB_test_files
    elif dataset_name == 'CALB' and type == 'train' and random_seed == 2024:
        return CALB_2024_train_files
    elif dataset_name == 'CALB' and type == 'vali' and random_seed == 2024:
        return CALB_2024_val_files
    elif dataset_name == 'CALB' and type == 'test' and random_seed == 2024:
        return CALB_2024_test_files
    elif dataset_name == 'CALB' and type == 'train' and random_seed == 42:
        return CALB_422_train_files
    elif dataset_name == 'CALB' and type == 'vali' and random_seed == 42:
        return CALB_422_val_files
    elif dataset_name == 'CALB' and type == 'test' and random_seed == 42:
        return CALB_422_test_files
    elif dataset_name == 'NA-ion' and type == 'train' and random_seed == 2021:
        return NAion_2021_train_files
    elif dataset_name == 'NA-ion' and type == 'vali' and random_seed == 2021:
        return NAion_2021_val_files
    elif dataset_name == 'NA-ion' and type == 'test' and random_seed == 2021:
        return NAion_2021_test_files
    elif dataset_name == 'NA-ion' and type == 'train' and random_seed == 2024:
        return NAion_2024_train_files
    elif dataset_name == 'NA-ion' and type == 'vali' and random_seed == 2024:
        return NAion_2024_val_files
    elif dataset_name == 'NA-ion' and type == 'test' and random_seed == 2024:
        return NAion_2024_test_files
    elif dataset_name == 'NA-ion' and type == 'train' and random_seed == 42:
        return NAion_42_train_files
    elif dataset_name == 'NA-ion' and type == 'vali' and random_seed == 42:
        return NAion_42_val_files
    elif dataset_name == 'NA-ion' and type == 'test' and random_seed == 42:
        return NAion_42_test_files

def cal_loss(dataset_name, random_seed):
    dataset_name = str(dataset_name)
    random_seed = int(random_seed)

    vali_loss = 0
    test_loss = 0
    vali_sample = 0
    test_sample = 0
    train_vali_avg_list = []
    train_test_avg_list = []
    vali_data = []
    test_data = []
    total_seen_unseen_label = []
    if file.startswith(f'{dataset_name}'):
        if 'ISU-ILCC' in dataset_name:
            dataset_name = 'ISU_ILCC'
        elif 'UL-PUR' in dataset_name:
            dataset_name = 'UL_PUR'

        train_data_names = [i for i in find_dataset(dataset_name, random_seed, type='train')]
        vali_data_names = [i for i in find_dataset(dataset_name, random_seed, type='vali')]
        test_data_names = [i for i in find_dataset(dataset_name, random_seed, type='test')]
        label_data = pd.read_json(os.path.join(path, file), lines=True)

        if random_seed == 42 and dataset_name == 'ZN-coin':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_ZN42.json', lines=True)
        elif random_seed == 2024 and dataset_name == 'ZN-coin':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_ZN2024.json', lines=True)
        elif random_seed == 42 and dataset_name == 'CALB':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_CALB42.json', lines=True)
        elif random_seed == 2024 and dataset_name == 'CALB':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_CALB2024.json', lines=True)
        elif random_seed == 42 and dataset_name == 'NA-ion':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_NA42.json', lines=True)
        elif random_seed == 2021 and dataset_name == 'NA-ion':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_NA2021.json', lines=True)
        elif random_seed == 2024 and dataset_name == 'NA-ion':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_NA2024.json', lines=True)
        else:
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test.json', lines=True)

        train_data = []
        for train_name in train_data_names:
            if 'Tongji' in train_name:
                train_name = train_name.replace('--', '-#')
            
            if train_name not in label_data:
                # print('Missing label for train file: ', train_name)
                continue
            train_data.append(label_data[train_name].values[0])
        for vali_name in vali_data_names:
            if 'Tongji' in vali_name:
                vali_name = vali_name.replace('--', '-#')

            if vali_name not in label_data:
                # print('Missing label for vali file: ', vali_name)
                continue
            vali_data.append(label_data[vali_name].values[0])
        for test_name in test_data_names:
            if 'Tongji' in test_name:
                test_name = test_name.replace('--', '-#')

            if test_name not in label_data:
                # print('Missing label for test file: ', test_name)
                continue

            test_name = test_name.replace('-#', '--')
            label = seen_unseen_label[test_name].values
            if label == 'seen':
                total_seen_unseen_label.append(1)
            else:
                total_seen_unseen_label.append(0)

            test_name = test_name.replace('--', '-#')
            test_data.append(label_data[test_name].values[0])

        train_avg = np.average(train_data, axis=0)
        train_vali_avg_list = [train_avg] * len(vali_data)
        train_test_avg_list = [train_avg] * len(test_data)

        for vali in vali_data:
            vali_loss = vali_loss + abs(vali - train_avg)
        for test in test_data:
            test_loss = test_loss + abs(test - train_avg)
        vali_sample = len(vali_data)
        test_sample = len(test_data)

    return vali_loss, test_loss, vali_sample, test_sample, vali_data, test_data, train_vali_avg_list, train_test_avg_list, total_seen_unseen_label


if __name__ == '__main__' :
    path = './dataset/Life labels/'
    files_path = os.listdir(path)
    files = [i for i in files_path if i.endswith('.json')]

    total_test_loss = 0
    total_vali_loss = 0
    total_vali_sample = 0
    total_test_sample = 0
    total_vali_data = []
    total_test_data = []
    total_train_vali_avg_list = []
    total_train_test_avg_list = []
    total_seen_unseen_labels = []
    seen_alpha_acc1 = 0
    seen_alpha_acc2 = 0
    unseen_alpha_acc1 = 0
    unseen_alpha_acc2 = 0
    target_dataset = sys.argv[1]
    random_seed = sys.argv[2]

    for file in tqdm(files):
        if target_dataset == 'MIX_large':
            if 'ZN-coin' in file:
                continue
            elif 'CALB' in file:
                continue
            elif 'NA-ion' in file:
                continue
            dataset_name = file.split('_')[0]
            if 'total' in file:
                dataset_name = 'total_MICH'
        elif target_dataset == 'MICH' or target_dataset == 'MICH_EXP':
            print('Please use [total_MICH] to evaluate.')
            break
        elif target_dataset == 'total_MICH':
            dataset_name = 'total_MICH'
        elif target_dataset == 'UL_PUR':
            print('UL_PUR has no enough samples to evaluate, try another dataset.')
            break
        elif target_dataset in file:
            dataset_name = file.split('_')[0]
        elif target_dataset == 'ISU-ILCC' or target_dataset == 'ISU_ILCC':
            dataset_name = 'ISU_ILCC'
        elif target_dataset == 'NAion':
            dataset_name = 'NA-ion'
        else:
            continue

        vali_loss, test_loss, vali_sample, test_sample, vali_data, test_data, train_vali_avg_list, train_test_avg_list, total_seen_unseen_label = cal_loss(dataset_name, random_seed)
        total_test_loss = total_test_loss + test_loss
        total_vali_loss = total_vali_loss + vali_loss
        total_test_sample = total_test_sample + test_sample
        total_vali_sample = total_vali_sample + vali_sample
        total_vali_data = total_vali_data + vali_data
        total_test_data = total_test_data + test_data
        total_train_vali_avg_list = total_train_vali_avg_list + train_vali_avg_list
        total_train_test_avg_list = total_train_test_avg_list + train_test_avg_list
        total_seen_unseen_labels = total_seen_unseen_labels + total_seen_unseen_label

        # print(f'Dataset {dataset_name}: Vali loss: {vali_loss} | Vali Sample: {vali_sample} | Test loss: {test_loss} | Test sample: {test_sample}')
    
    if target_dataset != 'UL_PUR' and target_dataset != 'MICH' and target_dataset != 'MICH_EXP':
        vali_mae = mean_absolute_error(total_vali_data, total_train_vali_avg_list)
        test_mae = mean_absolute_error(total_test_data, total_train_test_avg_list)
        vali_mape = mean_absolute_percentage_error(total_vali_data, total_train_vali_avg_list)
        test_mape = mean_absolute_percentage_error(total_test_data, total_train_test_avg_list)
        vali_mse = np.square(np.subtract(total_vali_data, total_train_vali_avg_list)).mean()
        vali_rmse = math.sqrt(vali_mse)
        test_mse = np.square(np.subtract(total_test_data, total_train_test_avg_list)).mean()
        test_rmse = math.sqrt(test_mse)

        hit_num = 0
        for pred, refer in zip(total_test_data, total_train_test_avg_list):
            relative_error = abs(pred - refer) / refer
            if relative_error <= 0.15:
                hit_num += 1
        alpha_acc1 = hit_num / len(total_train_test_avg_list) * 100

        indices_seen = [index for index, value in enumerate(total_seen_unseen_labels) if value == 1]
        indices_unseen = [index for index, value in enumerate(total_seen_unseen_labels) if value == 0]
        seen_preds = [total_test_data[i] for i in indices_seen]
        unseen_preds = [total_test_data[i] for i in indices_unseen]
        seen_references = [total_train_test_avg_list[i] for i in indices_seen]
        unseen_references = [total_train_test_avg_list[i] for i in indices_unseen]
        # seen
        hit_num = 0
        for pred, refer in zip(seen_preds, seen_references):
            relative_error = abs(pred - refer) / refer
            if relative_error <= 0.15:
                hit_num += 1
        if len(seen_references) == 0:
            seen_alpha_acc1 = 'No seen sample'
        else:
            seen_alpha_acc1 = hit_num / len(seen_references) * 100


        for pred, refer in zip(seen_preds, seen_references):
            relative_error = abs(pred - refer) / refer
            if relative_error <= 0.1:
                hit_num += 1
        if len(seen_references) == 0:
            seen_alpha_acc2 = 'No seen sample'
        else:
            seen_alpha_acc2 = hit_num / len(seen_references) * 100


        # unseen
        hit_num = 0
        for pred, refer in zip(unseen_preds, unseen_references):
            relative_error = abs(pred - refer) / refer
            if relative_error <= 0.15:
                hit_num += 1
        if len(unseen_references) == 0:
            unseen_alpha_acc1 = 'No unseen sample'
        else:
            unseen_alpha_acc1 = hit_num / len(unseen_references) * 100

        for pred, refer in zip(unseen_preds, unseen_references):
            relative_error = abs(pred - refer) / refer
            if relative_error <= 0.1:
                hit_num += 1
        if len(unseen_references) == 0:
            unseen_alpha_acc2 = 'No unseen sample'
        else:
            unseen_alpha_acc2 = hit_num / len(unseen_references) * 100

        print(f'Dummy Model : \nTest MAE: {test_mae} | Test MAPE: {test_mape} | Test RMSE: {test_rmse} | Vali MAE: {vali_mae} | Vali MAPE: {vali_mape} | Vali RMSE: {vali_rmse} | 0.15_acc: {alpha_acc1} | 0.15_acc_seen: {seen_alpha_acc1} | 0.1_acc_seen {seen_alpha_acc2} | 0.15_acc_unseen: {unseen_alpha_acc1} | 0.1_acc_unseen {unseen_alpha_acc2}')
