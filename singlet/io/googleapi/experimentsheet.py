# vim: fdm=indent
'''
author:     Fabio Zanini
date:       16/01/17
content:    Google Sheets API for the experiment sheets (picogreen, dilutions,
            etc.)
'''
# Modules
import os
import sys
import numpy as np
import pandas as pd
import argparse

from singlecell.googleapi.googleapi import GoogleIOError, GoogleAPI


# Globals
sheet_ids = {
        'sandbox': '12gmc34T8TOLfUFD-r02IwOloDvxu-UUnMP5am0rSt54',
        '10017006': '1SAwbZugkEb92hWJexVDmAfynhtTVMew0VKdvrRmSznY',
        '10017008': '1EM5uh_Eq4xqxSdQWS3u1Pe5b0Z-FsdxPn35ePst4NC0',
        }


# Classes / functions
class ExperimentSheet(GoogleAPI):
    def __init__(self, expname):
        self.expname = expname
        super().__init__(sheet_ids[expname])

    def set_visual_picogreen_backgrounds(self, threshold=0.5):
        '''Set the backgrounds of the picogreen visual table'''
        data = self.get_data('picogreen visual')
        bkgs = self.get_backgrounds('picogreen visual')
        bkgs_new = []
        for iline, (data_line, bkg_line) in enumerate(zip(data, bkgs)):
            if not (iline % 17):
                bkgs_new.append(bkg_line)
                continue

            bkg_line_new = []
            for datum, bkg in zip(data_line[:24], bkg_line[:24]):
                # Standards/blanks are light blue
                if bkg == '#cee1f3':
                    bkg_line_new.append(bkg)
                    continue
                if float(datum) >= threshold:
                    bkg_line_new.append('#99f7a7')
                else:
                    bkg_line_new.append('#f5a281')
            if len(bkg_line) > 24:
                bkg_line_new.extend(bkg_line[24:])
            bkgs_new.append(bkg_line_new)

        self.set_backgrounds('picogreen visual', bkgs_new)

    def make_visual_qPCR(self):
        if 'qPCR visual' not in self.get_sheetnames().values():
            self.add_sheet('qPCR visual')

        data_raw = self.get_data('qPCR raw')
        header = data_raw[0]
        data_raw = data_raw[1:]

        plateCol = header.index('Plate')
        wellCol = header.index('Well name')
        concCol = header.index('Quantity')
        taskCol = header.index('Task')
        detectorCol = header.index('Detector Name')

        platesUnique = sorted(set(row[plateCol] for row in data_raw))

        # Prepare output structure
        data = []
        for iPlate, plate in enumerate(platesUnique):
            data.append([plate])
            data.extend([[{'ACTB': '0', self.virus: '0'} for icol in range(24)] for irow in range(16)])

        # Fill output structure
        for row in data_raw:
            iPlate = platesUnique.index(row[plateCol])
            conc = row[concCol]
            detector = 'ACTB' if 'ACTB' in row[detectorCol] else self.virus
            task = row[taskCol]
            (row, col) = self.convert_A24_to_row_col(row[wellCol])

            if task != 'Unknown':
                datum = '-'
            elif '<' in conc:
                datum = 0
            else:
                datum = float(conc)

            data[iPlate * 17 + row + 1][col][detector] = datum

        # Add backgrounds
        bkgs = []
        for irow, row in enumerate(data):
            bkg_row = []
            if (irow % 17):
                for d in row:
                    if '-' in d.values():
                        bkg = '#222222'
                    elif (d['ACTB'] < 0.01) and (d[self.virus] < 0.001):
                        bkg = '#EBDB5B'
                    elif (d['ACTB'] > 0.01) and (d[self.virus] < 0.001):
                        bkg = '#B2DCFE'
                    elif (d['ACTB'] < 0.01) and (d[self.virus] > 0.001):
                        bkg = '#EB845B'
                    else:
                        bkg = '#A3FE8F'
                    bkg_row.append(bkg)
            bkgs.append(bkg_row)

        # Format output structure
        for irow, row in enumerate(data):
            if not (irow % 17):
                continue
            data[irow] = []
            for d in row:
                for detector, val in d.items():
                    if val == 0:
                        d[detector] = '-oo'
                    elif val != '-':
                        d[detector] = str(int(np.round(np.log10(val))))
                data[irow].append(d[self.virus]+' / '+d['ACTB'])

        # Add legend
        data.extend([
            [],
            ['Legend:'],
            ['log10qPCR('+self.virus+')/log10qPCR(ACTB)'],
            ['', 'Standards'],
            ['', 'ACTB < 0.01 and '+self.virus+' < 0.001'],
            ['', 'ACTB > 0.01 and '+self.virus+' < 0.001'],
            ['', 'ACTB < 0.01 and '+self.virus+' > 0.001'],
            ['', 'ACTB > 0.01 and '+self.virus+' > 0.001'],
            ])
        bkgs.extend([
            [],
            [],
            [],
            ['#222222'],
            ['#EBDB5B'],
            ['#B2DCFE'],
            ['#EB845B'],
            ['#A3FE8F'],
            ])

        self.set_sheet('qPCR visual', data)
        self.set_dimension_size('qPCR visual', 'columns', 53)
        self.set_backgrounds('qPCR visual', bkgs)

    def combine_picogreen_qPCR_visuals(self):
        import re

        picogreen = self.get_data('picogreen visual')
        qPCR = self.get_data('qPCR visual')

        if 'combined visual' not in self.get_sheetnames().values():
            self.add_sheet('combined visual')

        # Format data
        plates = [re.split('[_ -]', p[0])[0]
                  for irow, p in enumerate(picogreen) if not (irow % 17)]
        picogreenDict = {p: [row[:24]
                             for row in picogreen[iP * 17 + 1: iP * 17 + 17]]
                         for iP, p in enumerate(plates)}
        platesqPCR = [p[0] for irow, p in enumerate(qPCR)
                      if (not (irow % 17)) and len(p) and self.expname in p[0]]
        qPCRDict = {p: [row[:24] for row in qPCR[iP * 17 + 1: iP * 17 + 17]]
                    for iP, p in enumerate(platesqPCR)}

        # Copy structure of old and fill it
        from copy import deepcopy
        data = []
        for plate in plates:
            datum = deepcopy(picogreenDict[plate])
            if plate not in qPCRDict:
                for irow, row in enumerate(datum):
                    for icol, cell in enumerate(row):
                        datum[irow][icol] = '{:1.2f}'.format(float(cell))+'/-/-'
            else:
                datumqPCR = qPCRDict[plate]
                for irow, row in enumerate(datum):
                    for icol, cell in enumerate(row):
                        cellqPCR = datumqPCR[irow][icol].replace(' ', '')
                        datum[irow][icol] = '{:1.2f}'.format(float(cell))+'/'+cellqPCR
            data.append([plate])
            data.extend(datum)

        # Add backgrounds
        bkgs = []
        for irow, row in enumerate(data):
            bkg_row = []
            plate = plates[irow // 17]
            if (irow % 17):
                for icol, d in enumerate(row):
                    if plate not in platesqPCR:
                        field = d.split('/')[0]
                        # FIXME: extend to variable number of standards?
                        if (field.strip() == '-') or ((icol == 23) and (irow % 17 in (12, 13, 14, 15, 16))):
                            bkg = '#222222'
                        else:
                            d = {'PG': float(field)}
                            if d['PG'] < 1:
                                bkg = '#9B8270'
                            else:
                                bkg = '#B2DCFE'
                    else:
                        fields = d.split('/')
                        if '-' in fields:
                            bkg = '#222222'
                        else:
                            d = {'PG': float(fields[0]),
                                 self.virus: 10**float(fields[1]) if '-oo' not in fields[1] else 0,
                                 'ACTB':  10**float(fields[2]) if '-oo' not in fields[2] else 0,
                                 }
                            if d['PG'] < 1:
                                bkg = '#9B8270'
                            elif (d['ACTB'] < 0.01) and (d[self.virus] < 0.001):
                                bkg = '#EBDB5B'
                            elif (d['ACTB'] > 0.01) and (d[self.virus] < 0.001):
                                bkg = '#B2DCFE'
                            elif (d['ACTB'] < 0.01) and (d[self.virus] > 0.001):
                                bkg = '#EB845B'
                            else:
                                bkg = '#A3FE8F'
                    bkg_row.append(bkg)
            bkgs.append(bkg_row)

        # Add legend
        data.extend([
            [],
            ['Legend:'],
            ['PG[ng/ul]/log10qPCR('+self.virus+')/log10qPCR(ACTB)'],
            ['', 'Standards'],
            ['', 'PG < 1'],
            ['', 'PG > 1 and ACTB < 0.01 and '+self.virus+' < 0.001'],
            ['', 'PG > 1 and ACTB > 0.01 and '+self.virus+' < 0.001'],
            ['', 'PG > 1 and ACTB < 0.01 and '+self.virus+' > 0.001'],
            ['', 'PG > 1 and ACTB > 0.01 and '+self.virus+' > 0.001'],
            ])
        bkgs.extend([
            [],
            [],
            [],
            ['#222222'],
            ['#9B8270'],
            ['#EBDB5B'],
            ['#B2DCFE'],
            ['#EB845B'],
            ['#A3FE8F'],
            ])

        self.set_sheet('combined visual', data)
        self.set_dimension_size('combined visual', 'columns', 74)
        self.set_backgrounds('combined visual', bkgs)


# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyze experiment Google sheet')
    parser.add_argument('expname',
                        help='Experiment name')
    parser.add_argument('--make-visual-qPCR', action='store_true',
                        help='Analyze the raw qPCR table')
    parser.add_argument('--combine-picogreen-qPCR', action='store_true',
                        help='Analyze the raw qPCR table')
    
    args = parser.parse_args()

    exp = ExperimentSheet(args.expname)

    if args.make_visual_qPCR:
        exp.make_visual_qPCR()

    if args.combine_picogreen_qPCR:
        data = exp.combine_picogreen_qPCR_visuals()
