import sys
sys.path.append('../')

import tomli
import dataLogger
from pandas import json_normalize
import pandas as pd
from scipy.spatial import distance
from scipy.constants import c
import plotly.graph_objects as go
import numpy as np
#import autograd.numpy as np
#from autograd import grad, hessian
from scipy.optimize import approx_fprime
from scipy.optimize import fsolve, minimize
import itertools
import scipy.io as sio
import math
from tqdm import tqdm
import json
import operator
import argparse
import os
import time
from typing import List

pd.options.mode.chained_assignment = None  # default='warn'

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def s2m(time):
    '''seconds to meters'''
    return time * c

def m2s(distance):
    '''meters to seconds'''
    return distance / c

def gdop_calculation(position, coordinates):
    '''
        gdop calculation using the position and the set of anchors
    '''
    result = {"hdop":None, "vdop":None, "pdop":None, "tdop":None, "gdop":None}

    try:
        if len(coordinates) > 0 and len(coordinates) > 3:
            a_matrix = []
            x = position[0]
            y = position[1]
            z = position[2]
            for row in coordinates:
                r_matrix = math.sqrt(
                    (row[0] - x) ** 2 + (row[1] - y) ** 2 + (row[2] - z) ** 2)

                t_matrix = np.array([(np.array(row[0] - x) / r_matrix),
                                     (np.array(row[1] - y) / r_matrix),
                                     (np.array(row[2] - z) / r_matrix),
                                     1])
                a_matrix.append(t_matrix)

            # Test = np.cov(A)
            q_matrix = np.linalg.inv(
                np.matmul(np.transpose(a_matrix), a_matrix))
            # inv(A.' * A) * A.' * b
            # print(Q)

            vdop = round(math.sqrt(q_matrix[2, 2]), 2)
            hdop = round(math.sqrt(q_matrix[0, 0]) + math.sqrt(q_matrix[1, 1]), 2)
            pdop = round(
                math.sqrt(q_matrix[0, 0] + q_matrix[1, 1] + q_matrix[2, 2]), 2)
            tdop = round(math.sqrt(q_matrix[3, 3]), 2)
            # gdop = round(math.sqrt(pow(pdop, 2) + pow(tdop, 2)), 2)
            T = np.trace(q_matrix)
            G = np.sqrt(T)
            # gdop = round(math.sqrt(pow(pdop, 2) + pow(tdop, 2)), 2)
            gdop = G
            result = {"hdop":hdop, "vdop":vdop, "pdop":pdop, "tdop":tdop, "gdop":gdop}

        return result

    except:
        return result

class Positioner:
    def __init__(self, pklfile = None, jsonfile = None):
        self.subDfTime = {}
        self.subDfTimeComparison = {}
        self.pairsDfs = {}
        self.receivers = {}
        self.method = 'nelder-mead'
        self.tdoa_window_list = []
        self.refAnchors = []

        self.anchor_orientation = {
            7.0: -45,
            8.0: 225,
            9.0: 135,
            10.0: -90,
            11.0: -90,
            12.0: 135,
            13.0: 45,
            14.0: 90,
            15.0: -90,
            16.0: 45
        }

        if pklfile is not None:
            self.dfTime = self.load_pkl_data(pklfile)
        elif jsonfile is not None:
            self.dfTime = self.load_json_data(jsonfile)
        else:
            print("Not possible to load data")
            return

        
        self.cleanDataframe()
        #self.dfTime = self.dfTime[self.dfTime['srcAddr'].isin([7, 8, 9, 16])]
        self.dfTime = self.dfTime.set_index(['PCtimestamp'])
        self.get_positions()
        self.remove_positions()
        self.calculatePosDict()
        self.addresses = list(self.dfTime.srcAddr.unique())
        self.dfTimeProcessing()

        self.fillSingleDataframe()
        self.fillPairDataframe()
        #self.ideal_tdoa_calc()

        self.distance_between_anchors()
        self.process_all_tdoas()

        # Positioning algorithms
        self.generate_centroid()
        self.get_tdoa_window()

    def load_pkl_data(self, pklfile):
        dlog = dataLogger.Importer(pklfile)
        dfPkl = pd.DataFrame.from_dict(json_normalize(dlog.to_dict()))
        return dfPkl

    def load_json_data(self, jsonfile):
        with open(jsonfile) as f:
            data = json.load(f)

        dfJSON = pd.json_normalize(data)
        self.originalData = data
        dfJSON['tstamp'] = pd.to_datetime(dfJSON['tstamp'], unit="s")
        # Say that is UTC and transform it to the europe time
        dfJSON['tstamp'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
        dfJSON['PCtimestamp'] = dfJSON['tstamp']

        return dfJSON

    def calculatePosDict(self):
        self.AnchorPositions = self.dfTime[["PosX", "PosY",	"PosZ", "srcAddr"]]
        self.AnchorPositions = self.AnchorPositions.drop_duplicates(subset=["srcAddr"])
        self.AnchorPositions.set_index("srcAddr", inplace=True)
        self.AnchorPositions['address'] = self.AnchorPositions.index
        # Create dictionary with positions only
        self.posDict = self.AnchorPositions.to_dict(orient="index")

        return

    def get_position(self, addr):
        return [self.posDict[addr]['PosX'],
                self.posDict[addr]['PosY'],
                self.posDict[addr]['PosZ']]

    def calculate_weight(self, pos, anchor_pos, anchor_addr):
        anchor_angle = self.anchor_orientation[anchor_addr]
        # Calculate angle between position and anchor
        angle = np.degrees(np.arctan2(pos[1] - anchor_pos[1], pos[0] - anchor_pos[0]))
        # Calculate relative angle difference
        relative_angle = angle - anchor_angle

        if relative_angle <= 0:
            relative_angle += 360
        
        growth_rate = 0.2
        # Check if the relative angle is within the desired range
        if 0 <= relative_angle < 90 or 270 <= relative_angle <= 360:
            weight = 1  # Good signal weight from 0 to 90
        elif 90 <= relative_angle < 180:
            # Signal gets worse from 90 to 180
            weight = 1 - 1 / (1 + math.exp(-growth_rate * (relative_angle - 135)))
        elif 180 < relative_angle <= 270:
            # Signal improves from 180 to 270
            weight = 1 / (1 + math.exp(-growth_rate * (relative_angle - 225)))
        return weight

    def compute_dop(self, pos, refStationAddr):
        # Initialize matrices A and W
        N = len(self.receivers) - 1 
        A = np.zeros((N, 2))
        W = np.zeros((N, N))
        # Compute A and W matrices
        k = 0
        for anchor_id, anchor_pos in self.receivers.items():
            if anchor_id != refStationAddr:  # Exclude reference anchor
                ri = self.distance(pos, anchor_pos[:2])
                r_ref = self.distance(pos, self.receivers[refStationAddr][:2])
                W[k,k] = self.calculate_weight(pos, anchor_pos, refStationAddr)
                A[k] = [(anchor_pos[0] - pos[0]) / ri - (self.receivers[refStationAddr][0] - pos[0]) / r_ref,
                        (anchor_pos[1] - pos[1]) / ri - (self.receivers[refStationAddr][1] - pos[1]) / r_ref]
                k += 1
        # Compute DOP
        DOP = np.sqrt(np.trace(np.linalg.inv(A.T  @  A)))
        return DOP

    def plot_dop_contours(self, grid_size=25, num_points=100):
        # Initialize grid and DOP values
        x_vals = np.linspace(-5, grid_size, num_points)
        y_vals = np.linspace(-5, grid_size, num_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        DOP_values = np.zeros_like(X)

        # Compute DOP values for each point in the grid
        for i in range(num_points):
            for j in range(num_points):
                pos = [X[i, j], Y[i, j]]
                DOP_values[i, j] = self.compute_dop(pos, 0x07)
        
        DOP_values[DOP_values > 5] = 0
        # Create trace for scatter plot of anchor positions
        anchor_traces = []
        for anchor_id, pos in self.receivers.items():
            anchor_traces.append(go.Scatter(x=[pos[0]], y=[pos[1]], mode='markers', marker=dict(color='blue', size=10), name=f'Receiver {anchor_id}'))

            line_len = 2
            angle_deg = self.anchor_orientation[anchor_id]
            angle_rad = np.radians(angle_deg)
            ax = pos[0] + line_len * np.cos(angle_rad)
            ay = pos[1] + line_len * np.sin(angle_rad)
            if angle_deg == 90:
                arrow_symbol = 'triangle-up'
            elif angle_deg == -90:
                arrow_symbol = 'triangle-down'
            else:
                arrow_symbol = 'triangle-right' if angle_deg >= -90 and angle_deg <= 90 else 'triangle-left'
            arrow = go.Scatter(
                x=[pos[0], ax], 
                y=[pos[1], ay], 
                mode='lines+markers', 
                line=dict(color='black', width=3), 
                showlegend=False, 
                marker=dict(color='black', size=10, symbol=arrow_symbol)
            )
            anchor_traces.append(arrow)
        # Create trace for contour plot of DOP values
        contour_trace = go.Contour(x=x_vals, y=y_vals, z=DOP_values, colorscale='Reds', contours=dict(showlabels=True))

        # Define layout
        layout = go.Layout(
            title='DOP Contour Lines',
            xaxis=dict(title='X Position', showgrid=True),
            yaxis=dict(title='Y Position', showgrid=True),
            legend=dict(orientation='h', x=0, y=1.1, itemsizing='trace'),    
            height=600,
            width=800
        )

        # Plot
        fig = go.Figure(data=[contour_trace] + anchor_traces, layout=layout)
            
        fig.show()

    def get_hyperbolas(self, addr1:int, addr2:int, t):
        receivesat1 = np.array(self.get_position(addr1))
        receivesat2 = np.array(self.get_position(addr2))
        TDOA = t*c


        # Calculate distances between receivers
        d = np.linalg.norm(receivesat1 - receivesat2)

        # Calculate hyperbola parameters for Receivers
        a = TDOA / 2
        b = np.sqrt(np.abs((d / 2) ** 2 - a ** 2))
        midpoint = (receivesat1 + receivesat2) / 2

        # Plot ideal hyperbola
        theta = np.arctan2(receivesat2[1] - receivesat1[1], receivesat2[0] - receivesat1[0])
        t = np.linspace(-2, 2, 100)
        x_hyperbola = midpoint[0] + a * np.cosh(t) * np.cos(theta) - b * np.sinh(t) * np.sin(theta)
        y_hyperbola = midpoint[1] + a * np.cosh(t) * np.sin(theta) + b * np.sinh(t) * np.cos(theta)
        return x_hyperbola, y_hyperbola

    def distance_between_anchors(self):
        self.an2an_m = {}
        self.an2an_s = {}
        for combi in self.combi:
            self.an2an_m[combi] = \
            distance.euclidean(self.get_position(combi[0]), self.get_position(combi[1]))
            self.an2an_s[combi] = m2s(self.an2an_m[combi])
        return self.an2an_s

    def cleanDataframe(self):

        self.dfTime = self.dfTime.loc[(self.dfTime['type'] == "uwbMsgNotification") | \
                     (self.dfTime['type'] == "posMsgNotification")]

        self.dfTime.columns = self.dfTime.columns.str.replace(r'uwb.','')
        self.dfTime.columns = self.dfTime.columns.str.replace(r'content.','')
        self.dfTime.columns = self.dfTime.columns.str.replace(r'header.','')
        self.dfTime.columns = self.dfTime.columns.str.replace(r'payload.','')
        self.dfTime.columns = self.dfTime.columns.str.replace(r'diagnosis.','')
        self.dfTime.columns = self.dfTime.columns.str.replace(r'bcn','')

    def dfTimeProcessing(self):
        self.dfTime['index'] = self.dfTime.index

    def generateSingleDataframe(self, address):
        self.subDfTime[address] = self.dfTime[self.dfTime["srcAddr"] == address]

        self.subDfTime[address].reset_index(inplace=True,)
        self.subDfTime[address]['timestampTx_s'] = self.subDfTime[address]['TimestampTx']/(128*499.2e6)
        self.subDfTime[address]['timestampRx_s'] = self.subDfTime[address]['timestampRx']/(128*499.2e6)

        self.subDfTime[address]['timestampTxUnwrapped'] = np.unwrap(self.subDfTime[address]['TimestampTx'], period=(1 << 40))/(128*499.2e6)
        self.subDfTime[address]['timestampRxUnwrapped'] = np.unwrap(self.subDfTime[address]['timestampRx'], period=(1 << 40))/(128*499.2e6)

        self.subDfTime[address]['timestampTxUnwrappedDiff'] = self.subDfTime[address]['timestampTxUnwrapped'].diff()
        self.subDfTime[address]['timestampRxUnwrappedDiff'] = self.subDfTime[address]['timestampRxUnwrapped'].diff()
        self.subDfTime[address]['txUnwrap-rxUnwrap'] = self.subDfTime[address]["timestampTxUnwrapped"] - self.subDfTime[address]["timestampRxUnwrapped"]

        self.subDfTime[address]['txUnwrapDiff-rxUnwrapDiff'] = self.subDfTime[address]['timestampTxUnwrappedDiff'] - self.subDfTime[address]['timestampRxUnwrappedDiff']

        self.subDfTime[address]['txUnwrap-rxUnwrap_diff'] = self.subDfTime[address]['txUnwrap-rxUnwrap'].diff()

        self.subDfTime[address]['CFO'] = self.subDfTime[address]['timestampRxUnwrappedDiff']/self.subDfTime[address]['timestampTxUnwrappedDiff'] - 1
        self.subDfTime[address]['index'] = self.subDfTime[address].index

    def fillSingleDataframe(self):
        for address in self.addresses:
            self.generateSingleDataframe(address)

        # Post Processing
        for address in self.addresses:
            tempAddresses = self.addresses[:]
            tempAddresses.remove(address)
            self.subDfTimeComparison[address] = pd.DataFrame()
            for compaAddr in tempAddresses:
                subDfTimeCompare = self.subDfTime[address]['CFO'] - self.subDfTime[compaAddr]['CFO']
                self.subDfTimeComparison[address][f'CFO_comp_{compaAddr}'] = subDfTimeCompare


    def ideal_tdoa_calc(self):
        # Calculate distance between tag and anchor
        reference = self.ref_from_tag()
        self.idealDist2Tag_m = {}
        self.idealDist2Tag_s = {}

        self.idealTDOA = {}
        for addr in self.addresses:
            self.idealDist2Tag_s[addr] = \
                m2s(distance.euclidean(self.get_position(addr), reference))
            self.idealDist2Tag_m[addr] = \
                distance.euclidean(self.get_position(addr), reference)
        # Calculate TDOA for anchor pairs
        for combi in self.combi:
            self.idealTDOA[combi] = self.idealDist2Tag_s[combi[0]] - self.idealDist2Tag_s[combi[1]]

        return self.idealTDOA

    def process_all_tdoas(self):
        self.tdoaDfs = {}
        self.allTDoAsDf = pd.DataFrame()

        for addressPair in tqdm(self.permutations, desc="Calculating TDOA test"):
            addressA, addressB = addressPair
            tdoaDf_pair =self.calculate_tdoa(addressA,addressB)
            self.tdoaDfs[addressPair] = tdoaDf_pair
            
            if tdoaDf_pair.empty:
                continue
            else:
                self.allTDoAsDf = pd.concat([self.allTDoAsDf, tdoaDf_pair]).sort_index()

        # Force that we are really causal
        #    If virtualTime (middle point in TDOA) is bigger than
        #    realTime (received Timestamp), we are in the future
        self.allTDoAsDf = self.allTDoAsDf[(self.allTDoAsDf.realTime - self.allTDoAsDf.VirtualTime)>=0]
        return self.allTDoAsDf


    def calculate_tdoa(self,addressA,addressB):
        maximumTimeAllowedCFO = 0.7 # seconds
        maximumTimeAllowedTDOA = 0.7 # seconds

        RXwrapJumpTime = 2**40/(128*499.2e6)
        TXwrapJumpTime = 2**48/(128*499.2e6)

        dfA = self.subDfTime[addressA]
        dfB = self.subDfTime[addressB]

        dfA.set_index("timestampTx_s", inplace=True, drop = False)
        dfB.set_index("timestampTx_s", inplace=True, drop = False)

        dfAB = pd.concat([dfA, dfB]).sort_index()

        dataFrameTDOA = {}

        pcTimestamp = []
        tdoas = []
        VirtualTimes = []
        VirtualTimeDiffs = []
        cfoAs = []
        bufferSizeAs = []
        bufferSizeBs =  []
        realTime = []

        for i in range(1,len(dfA)):
            cfoA = None
            diffCFO = dfA['timestampTx_s'].iloc[i] - dfA['timestampTx_s'].iloc[i - 1]
            # RX check too? As should be similar, don't check it for the moment
            if  diffCFO < maximumTimeAllowedCFO:
                diffTxA = dfA['timestampTx_s'].iloc[i] - dfA['timestampTx_s'].iloc[i - 1]
                diffRxA = dfA['timestampRx_s'].iloc[i] - dfA['timestampRx_s'].iloc[i - 1]
                if diffRxA < -10:
                    diffRxA = dfA['timestampRx_s'].iloc[i] - dfA['timestampRx_s'].iloc[i - 1] + RXwrapJumpTime
                if diffRxA > 10:
                    diffRxA = dfA['timestampRx_s'].iloc[i] - dfA['timestampRx_s'].iloc[i - 1] - RXwrapJumpTime
                cfoA = diffRxA/diffTxA -1

                bufferTimestamps = dfAB.loc[(dfAB.index <= dfA['timestampTx_s'].iloc[i]) &
                                (dfAB.index >=dfA['timestampTx_s'].iloc[i]  - maximumTimeAllowedTDOA)]

                if (len(bufferTimestamps) < 2):
                    # print(f"stop! no data in time window {i}")
                    continue

                A = bufferTimestamps.loc[bufferTimestamps.srcAddr == addressA]
                B = bufferTimestamps.loc[bufferTimestamps.srcAddr == addressB]
                bufferSizeA = len(A)
                bufferSizeB = len(B)

                if len(A)==0 or len(B)==0:
                    # print(f"Length of one address is zero @ index {i}")
                    continue

                A_TX = A['timestampTx_s'].iloc[-1]
                B_TX = B['timestampTx_s'].iloc[-1]
                A_RX = A['timestampRx_s'].iloc[-1]
                B_RX = B['timestampRx_s'].iloc[-1]

                rxABdiff = A_RX - B_RX
                if rxABdiff >16:
                    rxABdiff = rxABdiff - RXwrapJumpTime
                if rxABdiff <-16:
                    rxABdiff = rxABdiff + RXwrapJumpTime

                tdoa = rxABdiff - (A_TX - B_TX)*(1+ cfoA)

                # Just place this TDOA reference time in the middle of the pair
                # Taking only A timestamp
                VirtualTimeTdoa = A_TX
                # Difference for statistic purpose
                VirtualTimeDiff = (A_TX - B_TX)
                pcTimestamp.append(A['PCtimestamp'].iloc[-1])
                tdoas.append(tdoa)
                VirtualTimes.append(VirtualTimeTdoa)
                cfoAs.append(cfoA)
                VirtualTimeDiffs.append(VirtualTimeDiff)
                bufferSizeAs.append(bufferSizeA)
                bufferSizeBs.append(bufferSizeB)
                realTime.append(A_TX)
            else:
                # print(dfA['timestampTx_s'].iloc[i])
                # print(f"too far way for CFO @ index {i}")
                pass

        dataFrameTDOA = {"tdoa": tdoas,
                        "PCtimestamp": pcTimestamp,
                        "VirtualTime": VirtualTimes,
                        "cfoA": cfoAs,
                        "VirtualTimeDiff": VirtualTimeDiffs,
                        "realTime": realTime,
                        "bufferSizeA": bufferSizeAs,
                        "bufferSizeB": bufferSizeBs,
                        "addresses": [(addressA,addressB)]*len(tdoas)}

        df = pd.DataFrame(data = dataFrameTDOA)
        df.set_index("realTime", inplace=True, drop = False)

        return df


    def plot_tdoas_over_time(self, yaxis_range =(-80e-9, 80e-9)):
        fig = go.Figure()
        for addresses in self.permutations:
            tdf = self.tdoaDfs[addresses]
            fig.add_trace(
                go.Scatter(
                    x = tdf["VirtualTime"],
                    y = tdf["tdoa"],
                    mode="markers+lines",
                    name=f"ADDR1: 0x{addresses[0]:02X}, ADDR2: 0x{addresses[1]:02X}",
                ))

        fig.update_layout(title="TDOA",
                        xaxis_title= "Time in Seconds (Tx timestamp)",
                        yaxis_title= "TDOA (in seconds)",
                        yaxis_range=yaxis_range)

        fig.show()

    def plot_positions(self):
        fig = go.Figure()
        for addr, pos in self.posDict.items():
            addr = int(addr)
            fig.add_trace(
                go.Scatter(
                        x=[pos['PosX']],  y = [pos['PosY']],
                        mode="markers",
                        name = f"Address: 0x{addr:02X}", ))

        fig.add_trace(
            go.Scatter(
                    x=self.TagPositions['position.x'],
                    y = self.TagPositions['position.y'],
                    mode="markers+lines",
                    name="Tag position",
        ))

        fig.update_layout(title="Tag positions calculated @ CHECKlet")
        fig.update_traces(marker_size=10)
        fig.update_layout(width=500, height=500)
        fig.show()



    def fillPairDataframe(self):
        self.addresses = sorted(set(self.addresses))
        #self.addresses = [i for i in self.addresses if i is not np.NaN]

        self.combi = list(itertools.combinations(self.addresses, 2))
        self.permutations = list(itertools.permutations(self.addresses, 2))


    def ref_from_tag(self):
        x = self.TagPositions.mean()['position.x']
        y = self.TagPositions.mean()['position.y']
        z = self.TagPositions.mean()['position.z']

        if (self.TagPositions.std()['position.x'] > 0.1) or \
            (self.TagPositions.std()['position.y'] > 0.1) or \
            (self.TagPositions.std()['position.z'] > 0.1):
            # print("warning! No static position given")
            pass

        return [x,y,z]



    def get_positions(self):

        self.TagPositions =self.dfTime.loc[~self.dfTime['meta.siteId'].isna()].copy()
        toKeep = ['PCtimestamp','position.x','position.y','position.z','position.cov','uwb.meta.siteId','position.cov.xx','position.cov.xy','position.cov.yx','position.cov.yy']
        columns = list(self.TagPositions.columns)
        toRemove = [column for column in columns if column not in toKeep]

        self.TagPositions = self.TagPositions.drop(toRemove, axis=1)
        self.TagPositions['index'] = self.TagPositions.index

        return self.TagPositions

    def remove_positions(self):
        self.dfTime = self.dfTime[self.dfTime['meta.siteId'].isna()]
        toRemove = ['position.x','position.y','position.z','meta.siteId','position.cov.xx','position.cov.xy','position.cov.yx','position.cov.yy']
        self.dfTime = self.dfTime.drop(toRemove, axis=1)


    def plotTogetherSingle(self, title="", x=None, y=None, rangeY = [None,None]):
        fig = go.Figure()
        for addr in self.addresses:
            tdf = self.subDfTime[addr]
            fig.add_trace(
                go.Scatter(
                    x=tdf.index if x is None else tdf[x],
                    y=tdf.index if y is None else tdf[y],
                    mode="markers+lines",
                    name=f"Address: 0x{addr:X}",
                ))

        fig.update_layout(title=title,
                        xaxis_title= x if x is not None else "",
                        yaxis_title= y if y is not None else "",
                        yaxis_range=rangeY)
        fig.update_layout()
        fig.show()


    def plotTogetherPair(self, title="", x=None, y=None, rangeY = [None,None]):
        fig = go.Figure()
        for addresses in self.combi:
            tdf = self.pairsDfs[addresses]
            fig.add_trace(
                go.Scatter(
                    x=tdf.index if x is None else tdf[x],
                    y=tdf.index if y is None else tdf[y],
                    mode="markers+lines",
                    name=f"Address1: 0x{addresses[0]:02X}, Address2: 0x{addresses[1]:02X}",
                ))

        fig.update_layout(title=title,
                        xaxis_title= x if x is not None else "",
                        yaxis_title= y if y is not None else "",
                        yaxis_range=rangeY)

        fig.show()

    def plotComparison(self, title="", x=None, y=None, rangeY = [None,None]):
        fig = go.Figure()
        for addr in self.addresses:
            tdf = self.subDfTimeComparison[addr]
            for column in tdf.columns:
                fig.add_trace(
                    go.Scatter(
                        y=tdf[column],
                        mode="markers+lines",
                        name=f"Address: 0x{addr:X} - {column}",
                ))

        fig.update_layout(title=title,
                        xaxis_title= x if x is not None else "",
                        yaxis_title= y if y is not None else "",
                        yaxis_range=rangeY)
        fig.update_layout()
        fig.show()


    def theoretical_tdoa_map(self):
        fig = go.Figure()
        fig.update_layout(title="Theoretical tdoa hyperbolas plot")

        # Plot  anchor positions
        for addr, pos in self.posDict.items():
            addr = int(addr)
            fig.add_trace(
                go.Scatter(
                        x=[pos['PosX']],  y = [pos['PosY']],
                        mode="markers",
                        name = f"Address: 0x{addr:}", ))
        # Plot tag position
        fig.add_trace(
            go.Scatter(
                    x=self.TagPositions['position.x'],
                    y = self.TagPositions['position.y'],
                    mode="markers+lines",
                    name="Tag position",
        ))

        # Plot hyperbola
        for tdoaAddr, tdoaValue in self.ideal_tdoa_calc().items():

            x, y = self.get_hyperbolas(tdoaAddr[0], tdoaAddr[1], tdoaValue)
            fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name =f"Addr1:0x{tdoaAddr[0]} Addr2:0x{tdoaAddr[1]} tdoa:{tdoaValue:2.3}",
                mode="lines"))
        fig.show()

    def plot_measured_tdoa_map(self, calculatedPoint = None, pairsUsed:List[List] = None, indexHyperbola = 50):

        fig = go.Figure()

        for addr, pos in self.posDict.items():
            addr = int(addr)
            fig.add_trace(
                go.Scatter(
                        x=[pos['PosX']],  y = [pos['PosY']],
                        mode="markers",
                        marker=dict( size=10, symbol="circle-x", line=dict(width=0.5,)),

                        name = f"Address: 0x{addr:02X}", ))


        # Plot tag position
        fig.add_trace(
            go.Scatter(
                    x=self.TagPositions['position.x'],
                    y = self.TagPositions['position.y'],
                    mode="markers+lines",
                    name="PP filtered Tag position",
                    marker=dict( size=3, symbol="x-thin", line=dict(width=1,)),
        ))

        if calculatedPoint is not None:
            # Plot tag position
            fig.add_trace(
            go.Scatter(
                    x=calculatedPoint[0],
                    y = calculatedPoint[1],
                    mode="markers+lines",
                    name="BST raw Tag position",
                    marker=dict( size=3, symbol="x-thin", line=dict(width=1,)),
        ))

        fig.update_layout( height=800, width=800, title="Measured tdoa hyperbolas plot")
        fig.show()

    def generate_centroid(self):
        for addr, values in self.posDict.items():
            self.receivers[addr] = [values['PosX'],values['PosY'], values['PosZ']]
        return self.centroid()

    def centroid(self):
        total_x = 0
        total_y = 0
        total_z = 0
        for pos in self.receivers.values():
            total_x += pos[0]
            total_y += pos[1]
            total_z += pos[2]
        centroid_x = total_x / len(self.receivers)
        centroid_y = total_y / len(self.receivers)
        centroid_z = total_z / len(self.receivers)
        return np.array([centroid_x, centroid_y, centroid_z])

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def get_tdoa_window(self):
        addresses_series = self.dfTime["srcAddr"]
        unique_values = []
        start_index = 1

        for i, value in enumerate(addresses_series):
            if value in unique_values:

                dt = self.dfTime["tstamp"].iloc[i] -  self.dfTime["tstamp"].iloc[start_index-1]
                self.tdoa_window_list.append(dt.total_seconds())
                self.refAnchors.append(unique_values[0])
                start_index += len(unique_values)
                
                # Restart from the duplicated value
                unique_values = [value]
                
            else:
                # Add the value to unique_values
                unique_values.append(value)
                
    def lse_method(self,
                   initial_guess,
                   tdoa_window,
                   minimize_method, 
                   iterations = 200):
    
        self.TDOA_measurement = tdoa_window
        result = minimize(self.calculate_total_error,
                            initial_guess,
                            method=minimize_method, 
                            jac = self.gradient,
                            #hess= self.calculate_total_error_hessian,
                            options={#'xtol': 1e-8,
                                     'disp': False,
                                     'maxiter':iterations,
                                     "return_all": True})
        return result

    def process_dataframe_positioning(self,
                                      algoType = "algorithmLSE",
                                      timeWindow = 0.25,
                                      timeBased=False):
        TDOADF = self.allTDoAsDf
        self.algoType = algoType

        positionsList = []
        timesList = []
        pcTimestampList = []
        TDoASsizeList = []
        measurementTimeSpan = TDOADF.index[-1] - TDOADF.index[0]
        resultsList = []
 
        C = self.generate_centroid()
        if timeBased:
            # Slots of time every 250ms
            total_steps = math.ceil(measurementTimeSpan/timeWindow)
        else:
            # Calculate for each new TDOA time
            #

            total_steps = len(self.tdoa_window_list)

            # indexes = TDOADF.index.unique()
            # total_steps = len(indexes)

        previous_t = 0
        previous_pos = 0
        for i in tqdm(range(1,total_steps), desc=f"Calculating positions {self.method}"):
            # Select time window that is going to be used

            if timeBased:
                start = timeWindow*(i-1)
                stop = timeWindow*i

                TDOA4Position = TDOADF.loc[(TDOADF.index > (TDOADF.index[0] + start))
                                        & (TDOADF.index < (TDOADF.index[0] + stop))]
            else:
                start = previous_t
                stop = self.tdoa_window_list[i-1] + start

                TDOA4Position = TDOADF.loc[(TDOADF.index >= (TDOADF.index[0] + start))
                                        & (TDOADF.index <= (TDOADF.index[0] + stop))]
                
                previous_t = stop

                # TDOA4Position = TDOADF.loc[(TDOADF.index > (indexes[i] - timeWindow))
                #                         & (TDOADF.index < (indexes[i]))]

            # Check if enough data is available to execute algorithm
            numberOfTDOAS = len(TDOA4Position)
            if numberOfTDOAS < 3:
                # TODO: Add empty list to signalize status
                continue

            # Current measurement time (latest TDOA timestamp)
            CurrentTime = TDOA4Position.index[-1]

            # Get dictionary for processing
            toPositionDict = TDOA4Position[['addresses','tdoa']].set_index("addresses").to_dict()['tdoa']

            addresses = [10, 11, 12, 13]
            toPositionDict = {pair: toPositionDict[pair] for pair in toPositionDict.keys() if any(addr in pair for addr in addresses)}
            # Process data with LSE method

            refStationAddr = 0x0A
            count = sum(1 for key in toPositionDict.keys() if refStationAddr in key)

            C = np.array([10.5, 3.5, 1.2])
            if count < 3:
                # TODO: Add empty list to signalize status
                continue

            if self.algoType == "algorithmLSE":
                positionResults = self.lse_method(C, toPositionDict, minimize_method = "BFGS")
                position = positionResults.x
            elif self.algoType == "algorithmCHAN":
                positionResults = self.chans_method(toPositionDict)
                if positionResults is None:
                    continue
                position = positionResults[:3].tolist()
            elif self.algoType =="algorithmTAYLOR":
                positionResults = self.taylor_series_method(C,toPositionDict, refStationAddr)
                if positionResults is None:
                    continue
                positionResults[2] =  1.2
                position = positionResults.tolist()

            elif self.algoType =="algorithmTAYLOR-CHAN":

                init_guess = self.chans_method(toPositionDict, refStationAddr)
                if init_guess is None:
                    continue

                init_guess = init_guess[:3]            
                init_guess[2] =  1.2  

                if distance.euclidean(C, init_guess) > 20:
                    init_guess = C
                elif distance.euclidean(C, init_guess) < 15:
                    C = init_guess

                positionResults = self.taylor_series_method(C, toPositionDict, refStationAddr)
                if positionResults is None:
                    continue

                positionResults[2] =  1.2  
                if distance.euclidean(C, positionResults) < 5:
                    C = positionResults
                position = positionResults.tolist()
            else:
                print("No algorithm selected")
                return None

            # Check if solution is too divergent to take it into consideration
            # if np.any(position) != None and (abs(position[0]) > 100 or abs(position[1]) > 100):
            #     continue
                
            
            if position is not None and distance.euclidean(C, position) > 30:
               continue
            elif position is not None and distance.euclidean(C, position) < 5:
               C = position


            # Distance between points consecutive points is not that far way, use it
            # next as the start point for the next minimization cycle.
            # if position is not None and distance.euclidean(C, position) < 10:
            #     C = position
            # else:
            #     C = self.centroid()


            # Just add al the results for this measurement and continue
            positionsList.append(position)
            timesList.append(CurrentTime)
            TDoASsizeList.append(len(TDOA4Position))
            resultsList.append(positionResults)
            pcTimestampList.append(TDOA4Position['PCtimestamp'].iloc[-1])

        # Loop over, just place things together
        positionDict = {"positions": positionsList,
                        "time": timesList,
                        "PCtimestamp": pcTimestampList,
                        "algoDetails.numTDoA": TDoASsizeList,
                        "algoDetails": resultsList}

        # Just move up the positions to x,y,z
        if positionsList:

            self.PositionDf = pd.DataFrame(positionDict)
            self.PositionDf['x'] = self.PositionDf["positions"].apply(pd.Series)[0].tolist()
            self.PositionDf['y'] = self.PositionDf["positions"].apply(pd.Series)[1].tolist()
            self.PositionDf['z'] = self.PositionDf["positions"].apply(pd.Series)[2].tolist()
            self.PositionDf = self.PositionDf.drop(columns=['positions'])

            # Take UWB time and use the first PC time to place it correctly in time
            timeDelta = pd.to_timedelta(self.PositionDf["time"] - self.PositionDf["time"][0] , unit="s")
            self.PositionDf['time'] = self.PositionDf['PCtimestamp'][0] + timeDelta
            # Generate timestamp from time too
            self.PositionDf['tstamp'] = self.PositionDf['time'].apply(pd.Timestamp.timestamp)

    def calculate_total_error(self,current_position):
        total_error = 0
        x, y, z = current_position 
   
        for (receiver1, receiver2), expected_tdoa in self.TDOA_measurement.items():
            receiver1_pos = self.receivers[receiver1]
            receiver2_pos = self.receivers[receiver2]
            distance1 = np.linalg.norm(np.array([x, y, z]) - receiver1_pos)
            distance2 = np.linalg.norm(np.array([x, y, z]) - receiver2_pos)
            actual_tdoa_distance = (distance1 - distance2)
            error = (actual_tdoa_distance - (expected_tdoa*c)) ** 2
            total_error += error
        return total_error

    def distance_error_sum(self, initial_pos, refStationAddr):
        """
        Sum of squared distance errors with anchoring to point (x0, y0, z0)
        """
        x, y, z = initial_pos
        x0, y0, z0 = self.receivers[refStationAddr]
        error_sum = 0

        TDOA_to_ref_station = self.get_refAnchor_tdoa(refStationAddr)
        for sat, measured_d in TDOA_to_ref_station.items():
            xi, yi, zi = self.receivers[sat]
            error_sum += ((measured_d*c) - np.sqrt((x - xi)**2 + (y - yi)**2 + (z - zi)**2) + np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2))**2
        return error_sum
    
    def gradient(self, initial_pos):
        """
        Gradient of the objective function with respect to x, y, and z
        """
        grad = approx_fprime(initial_pos, self.calculate_total_error, epsilon=1e-6)
        return grad

    def hessian(self, initial_pos):
        """
        Hessian of the objective function
        """
        hessian = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                hessian[i, j] = approx_fprime(initial_pos, lambda x: self.gradient(x)[i], epsilon=1e-6)[j]
        return hessian

    def get_refAnchor_tdoa(self, refStationAddr):

        TDOA_to_ref_station = {}
        for pair, value in self.TDOA_measurement.items():
            if refStationAddr in pair:
                other_station = pair[0] if pair[1] == refStationAddr else pair[1]
                TDOA_to_ref_station[other_station] = value
        return TDOA_to_ref_station
        
    def chans_method(self, tdoa_window, refStationAddr = 0x0B):
        '''No working for the moment: @Tomas to take a look'''
        self.TDOA_measurement = tdoa_window

        ref_x, ref_y, ref_z = self.receivers[refStationAddr]
        TDOA_to_ref_station = self.get_refAnchor_tdoa(refStationAddr)


        Ga = []
        h = []
        for sat in self.receivers.keys():
            if sat != refStationAddr and sat in TDOA_to_ref_station :
                xi, yi, zi = self.receivers[sat]
                x1, y1, z1 = self.receivers[refStationAddr]
                xi_diff, yi_diff, zi_diff = xi - ref_x, yi - ref_y, zi - ref_z
                Ga.append([xi_diff, yi_diff, zi_diff, TDOA_to_ref_station[sat]])
                h.append(0.5 * (TDOA_to_ref_station[sat]**2 - xi**2 - yi**2 - zi**2 + x1**2 + y1**2 + z1**2))

        Ga = np.array(Ga) * -1
        h = np.array(h)
        Q = np.eye(len(TDOA_to_ref_station))
        try:
            za_initial = np.linalg.inv(Ga.T @ np.linalg.inv(Q) @ Ga) @ Ga.T @ np.linalg.inv(Q) @ h

            Ri = [np.linalg.norm(za_initial[:3] - self.receivers[sat]) for sat in TDOA_to_ref_station.keys()] 
            B = np.diag(Ri)
            Psi = c**2 * (B @ Q @ B)

            za_initial = np.linalg.inv(Ga.T @ np.linalg.inv(Psi) @ Ga) @ Ga.T @ np.linalg.inv(Psi) @ h
        except np.linalg.LinAlgError:
            za_initial = None

        return za_initial
    
    def chans_method_1(self, tdoa_window):
        '''Chan's method for position estimation'''

        self.TDOA_measurement = tdoa_window

        Ga = []
        h = []
        
        # Constructing Ga matrix and h vector
        for pair, tdoa in self.TDOA_measurement.items():
            receiver1, receiver2 = pair
            x1, y1, z1 = self.receivers[receiver1]
            x2, y2, z2 = self.receivers[receiver2]
            
            xi_diff, yi_diff, zi_diff = x2 - x1, y2 - y1, z2 - z1
            Ga.append([xi_diff, yi_diff, zi_diff, tdoa])
            h.append(0.5 * (tdoa**2 - x2**2 - y2**2 - z2**2 + x1**2 + y1**2 + z1**2))

        Ga = np.array(Ga) * -1
        h = np.array(h)
        
        # Weighted least squares solution
        Q = np.eye(len(self.TDOA_measurement))
        try:
            za_initial = np.linalg.inv(Ga.T @ np.linalg.inv(Q) @ Ga) @ Ga.T @ np.linalg.inv(Q) @ h

            # Calculate the range estimates from the initial position estimate
            # Ri = [np.linalg.norm(za_initial[:3] - self.receivers[receiver]) for receiver in self.receivers]
            # B = np.diag(Ri)
            # Psi = c**2 * (B @ Q @ B)

            # Refine the position estimate using weighted least squares
            # za_initial = np.linalg.inv(Ga.T @ np.linalg.inv(Psi) @ Ga) @ Ga.T @ np.linalg.inv(Psi) @ h
        except np.linalg.LinAlgError:
            za_initial = None

        return za_initial

    def taylor_series_method(self, initial_guess,
                              tdoa_window,
                              refStationAddr = 0xB,
                              max_iterations=200,
                              tolerance=1e-6, ):
        '''No working for the moment: @Tomas to take a look'''

        self.TDOA_measurement = tdoa_window

        x0, y0, z0 = initial_guess

        TDOA_to_ref_station = self.get_refAnchor_tdoa(refStationAddr)

        #TDOA_to_ref_station = {key: value for key, value in TDOA_to_ref_station.items() if abs(value) < 3e-08}

        M = len(TDOA_to_ref_station)
        Q = np.eye(M)
        for _ in range(max_iterations):
            R_ref = np.linalg.norm(self.receivers[refStationAddr] - np.array([x0, y0, z0]))
            predicted_TDoA = {}
            Ri = {}
            for sat, pos in TDOA_to_ref_station.items():
                if sat != refStationAddr:
                    dist_ref = np.linalg.norm(self.receivers[refStationAddr] - np.array([x0, y0, z0]))
                    dist_sat = np.linalg.norm(self.receivers[sat] - np.array([x0, y0, z0]))

                    predicted_TDoA[sat] = dist_sat - dist_ref
                    Ri[sat] = dist_sat

            ht = np.array([ (tdoa* c) - predicted_TDoA[sat] for sat, tdoa in TDOA_to_ref_station.items()])
            Gt = np.zeros((M, 3))
            for i, receiver_i in enumerate(TDOA_to_ref_station.keys()):
                dTDoA_dx = ((self.receivers[refStationAddr][0] - x0) / R_ref) - ((self.receivers[receiver_i][0] - x0) / Ri[receiver_i])
                dTDoA_dy = ((self.receivers[refStationAddr][1] - y0) / R_ref) - ((self.receivers[receiver_i][1] - y0) / Ri[receiver_i])
                dTDoA_dz = ((self.receivers[refStationAddr][2] - z0) / R_ref) - ((self.receivers[receiver_i][2] - z0) / Ri[receiver_i])
                Gt[i, 0] = dTDoA_dx
                Gt[i, 1] = dTDoA_dy
                Gt[i, 2] = dTDoA_dz
            try:
                delta = np.linalg.inv(Gt.T @ np.linalg.inv(Q) @ Gt) @ Gt.T @ np.linalg.inv(Q) @ ht
            except np.linalg.LinAlgError:
                return None
            x_new = x0 + delta[0]
            y_new = y0 + delta[1]
            z_new = z0 + delta[2]

            if np.linalg.norm(delta) < tolerance:
                return np.array([x_new, y_new, z_new])
            x0, y0, z0 = x_new, y_new, z_new
        return np.array([x_new, y_new, z_new])

    def taylor_series_method_1(self, initial_guess,
                         tdoa_window,
                         max_iterations=200,
                         tolerance=1e-6):
        
        x0, y0, z0 = initial_guess

        M = len(tdoa_window)  # Number of TDOA pairs
        Q = np.eye(M)

        for _ in range(max_iterations):
            ht = np.zeros(M)
            Gt = np.zeros((M, 3))

            # Calculate predicted TDOA and update ht and Gt matrices
            for i, ((receiver_i, receiver_j), measured_TDOA) in enumerate(tdoa_window.items()):
                dist_i = np.linalg.norm(self.receivers[receiver_i] - np.array([x0, y0, z0]))
                dist_j = np.linalg.norm(self.receivers[receiver_j] - np.array([x0, y0, z0]))
                predicted_TDoA = dist_i - dist_j

                ht[i] = (measured_TDOA * c - predicted_TDoA)
                Gt[i, 0] = ((self.receivers[receiver_i][0] - self.receivers[receiver_j][0]) / dist_i - 
                            (self.receivers[receiver_i][0] - self.receivers[receiver_j][0]) / dist_j)
                Gt[i, 1] = ((self.receivers[receiver_i][1] - self.receivers[receiver_j][1]) / dist_i - 
                            (self.receivers[receiver_i][1] - self.receivers[receiver_j][1]) / dist_j)
                Gt[i, 2] = ((self.receivers[receiver_i][2] - self.receivers[receiver_j][2]) / dist_i - 
                            (self.receivers[receiver_i][2] - self.receivers[receiver_j][2]) / dist_j)

            try:
                delta = np.linalg.inv(Gt.T @ np.linalg.inv(Q) @ Gt) @ Gt.T @ np.linalg.inv(Q) @ ht
            except np.linalg.LinAlgError:
                return None
            
            x_new = x0 + delta[0]
            y_new = y0 + delta[1]
            z_new = z0 + delta[2]

            if np.linalg.norm(delta) < tolerance:
                return np.array([x_new, y_new, z_new])

            x0, y0, z0 = x_new, y_new, z_new

        return np.array([x_new, y_new, z_new])


    def export_json(self, filename):
        list2export1 = self.PositionDf.to_dict(orient="records")
        for element in list2export1:
            # Add data type
            element['type'] = self.algoType
            element['position'] = {'x':element['x'],'y': element['y'],'z': element['z']}
            del element['x']
            # restructure to get nice format
            element['extra'] = {'numTDoA': element['algoDetails.numTDoA']}
            element['extra'] = element['algoDetails']
            element['tstamp'] = element['time'].timestamp()
            del element['algoDetails.numTDoA']
            del element['algoDetails']
            del element['PCtimestamp']
            del element['time']

        list2export2 = self.allTDoAsDf.to_dict(orient="records")
        for element in list2export2:
            element['type'] = "TDoA"
            element['tstamp'] = element['PCtimestamp'].timestamp()
            del element['PCtimestamp']

        # Clean labels that not corespond to uwbMsgNotification or posMsgNotification
        list2export3 = []
        for item in self.originalData:
            if item['type'] == "uwbMsgNotification" or item['type'] == "posMsgNotification":
                list2export3.append(item)

        finalList = list2export1 + list2export2 + list2export3

        finalList = sorted(finalList, key=operator.itemgetter('tstamp'))
        with open(filename, 'w') as fp:
            json.dump(finalList, fp, cls=NumpyEncoder)

    def export_simplified_csv(self,filename):
        # reduce to x,y,z
        toExport = self.PositionDf.copy()
        toExport['t'] = toExport['tstamp']
        toExport[['t','x','y','z']].to_csv(filename, index=False)

    def export_matlab(self, filename):

        self.OutDataMat = {}
        self.OutDataMat['TagPositions'] = self.TagPositions.to_dict('list')
        self.OutDataMat['timestamps'] = self.dfTime.to_dict('list', )
        self.OutDataMat['AnchorPositions'] = self.AnchorPositions.to_dict('list', )
        sio.savemat(filename,self.OutDataMat)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Application to calculate from the timestamps received by the\
                                     pinpoint system, positions using different algorithms')
    parser.add_argument('--input', '-I', help = "JSON file from measured data")
    parser.add_argument('--output', '-O', required=False, default = "input", \
                        help = "Output filename. If nothing selected, input filename used")
    parser.add_argument('--mat', '-M', default=False,  action='store_true',\
                        help = "Use this option if mat file generation wanted")
    parser.add_argument('--csv', '-C', default=False, action='store_true',\
                        help = "Use this option if simplified CSV file generation wanted")
    parser.add_argument('--json', '-J', default=False, action='store_true',\
                        help='Generate complete dataset with all intermediate data used for \
                        derivation of positions (including TDOA and positioning algorithm intermediate steps)')

    args = parser.parse_args()
    filename_input = args.input
    filename_output = args.output
    matGeneration = args.mat
    jsonGeneration = args.json
    csvGeneration = args.csv
    # if nothing explicitly requested, default to JSON
    jsonGeneration = not matGeneration and not csvGeneration

    if filename_output =="input":
        filename_output = os.path.splitext(filename_input)[0]
    else:
        path2check = os.path.dirname(filename_output)
        if path2check != "" and not os.path.exists(path2check):
            os.makedirs(path2check)

    P= Positioner(jsonfile=filename_input)
    P.process_dataframe_positioning(algoType="algorithmLSE",
                                    timeWindow=0.2,
                                    timeBased=False)

    if jsonGeneration:
        f = filename_output
        if not f.endswith('.json'):
            f += '.json'
        P.export_json(f)
        print(f"File generated: {f}")
    if matGeneration:
        f = filename_output
        if not f.endswith('.mat'):
            f += '.mat'
        P.export_matlab(f)
        print(f"File generated: {f}")
    if csvGeneration:
        f = filename_output
        if not f.endswith('.csv'):
            f += '.csv'
        P.export_simplified_csv(f)
        print(f"File generated: {f}")
