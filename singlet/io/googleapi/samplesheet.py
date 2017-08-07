# vim: fdm=indent
'''
author:     Fabio Zanini
date:       16/01/17
content:    Google Sheets API for the sample sheet.
'''
# Modules
import os
import numpy as np
import pandas as pd

from .googleapi import GoogleIOError, GoogleAPI
from .filenames import googleapi_config


# Globals


# Classes / functions
class SampleSheet(GoogleAPI):
    def __init__(self, sheetname):
        sheetid = googleapi_config['samplesheets'][sheetname]['id']
        super().__init__(sheetid)

    def get_number_virus_reads(self, virus, icols=None):
        '''Get the number of virus reads from the spreadsheet'''
        sheetname = 'sequenced'
        vircolname = 'number'+virus.capitalize()+'Reads'

        if icols is None:
            icols = self.get_header_columns_indices(
                    ['name', 'experiment', vircolname],
                    sheetname)

        if 'name' not in icols:
            raise ValueError('name must be part of the icols dict')
        if vircolname not in icols:
            raise ValueError(vircolname+' must be part of the icols dict')

        # Ask for the name first, that determines the range
        colnames = ['name'] + [cn for cn in icols if cn !='name']

        # Get the values
        data = {}
        for colname in colnames:
            icol = icols[colname]

            # Google figures out the max row number
            if colname == 'name':
                rangeName = sheetname+'!'+icol+'2:'+icol+'100000'
            else:
                nNames = len(data['name'])
                rangeName = sheetname+'!'+icol+'2:'+icol+str(1+nNames)

            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheetId, range=rangeName).execute()
            values = result.get('values', [])

            # Google cuts trailing None
            if (colname != 'name') and (len(values) < nNames):
                values.extend([['']] * (nNames - len(values)))

            # format
            if colname == vircolname:
                values = np.array([int(v[0]) if (len(v) and (v[0] != '')) else -1 for v in values], int)
            else:
                values = np.array(values)[:, 0]

            data[colname] = values

        # Check consistency
        l = len(data['name'])
        for colname, datum in data.items():
            if len(datum) != l:
                raise ValueError('Not all columns have the same length')

        data = pd.DataFrame(data)
        return data

    def set_number_virus_reads(self, virus, number_virus_reads):
        '''Set the number of virus reads in the spreadsheet

        Arguments:
           - virus (str): the name of the virus
           - number_virus_reads (pandas.Series): the new numbers to update. The
             series must be indexed by sample name (e.g. 1001700501_A4)
        '''
        sheetname = 'sequenced'
        vircolname = 'number'+virus.capitalize()+'Reads'

        icols = self.get_header_columns_indices(
                ['name', vircolname],
                sheetname)

        data = self.get_number_virus_reads(virus, icols=icols)
        names = data['name'].tolist()
        number_reads = data[vircolname].values

        # Set new values
        for name, n_reads in number_virus_reads.iteritems():
            number_reads[names.index(name)] = n_reads

        # Upload
        rangeName = sheetname+'!'+icols[vircolname]+'2:'+str(2+data.shape[0])
        body = {'majorDimension': 'ROWS',
                'values': [[str(v)] if v != -1 else [''] for v in number_reads]}
        result = self.service.spreadsheets().values().update(
            spreadsheetId=self.spreadsheetId,
            range=rangeName,
            body=body,
            valueInputOption='USER_ENTERED',
            ).execute()
        return result

    def get_number_dengue_reads(self, icols=None):
        '''Get the number of dengue reads from the spreadsheet'''
        sheetname = 'sequenced'

        if icols is None:
            icols = self.get_header_columns_indices(
                    ['name', 'experiment', 'numberDengueReads'],
                    sheetname)

        if 'name' not in icols:
            raise ValueError('name must be part of the icols dict')

        # Ask for the name first, that determines the range
        colnames = ['name'] + [cn for cn in icols if cn !='name']

        # Get the values
        data = {}
        for colname in colnames:
            icol = icols[colname]

            # Google figures out the max row number
            if colname == 'name':
                rangeName = sheetname+'!'+icol+'2:'+icol+'100000'
            else:
                nNames = len(data['name'])
                rangeName = sheetname+'!'+icol+'2:'+icol+str(1+nNames)

            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheetId, range=rangeName).execute()
            values = result.get('values', [])
            if not values:
                raise GoogleIOError('No data found for column '+colname+'.')

            # Google cuts trailing None
            elif (colname != 'name') and (len(values) < nNames):
                values.extend([['']] * (nNames - len(values)))

            # format
            if colname == 'numberDengueReads':
                values = np.array([int(v[0]) if (len(v) and (v[0] != '')) else -1 for v in values], int)
            else:
                values = np.array(values)[:, 0]

            data[colname] = values

        # Check consistency
        l = len(data['name'])
        for colname, datum in data.items():
            if len(datum) != l:
                raise ValueError('Not all columns have the same length')

        data = pd.DataFrame(data)
        return data

    def set_number_dengue_reads(self, number_dengue_reads):
        '''Set the number of dengue reads in the spreadsheet

        Arguments:
           - number_dengue_reads (pandas.Series): the new numbers to update. The
             series must be indexed by sample name (e.g. 1001700501_A4)
        '''
        sheetname = 'sequenced'

        icols = self.get_header_columns_indices(
                ['name', 'numberDengueReads'],
                sheetname)

        data = self.get_number_dengue_reads(icols=icols)
        names = data['name'].tolist()
        number_reads = data['numberDengueReads'].values

        # Set new values
        for name, n_reads in number_dengue_reads.iteritems():
            number_reads[names.index(name)] = n_reads

        # Upload
        rangeName = sheetname+'!'+icols['numberDengueReads']+'2:'+str(2+data.shape[0])
        body = {'majorDimension': 'ROWS',
                'values': [[str(v)] if v != -1 else [''] for v in number_reads]}
        result = self.service.spreadsheets().values().update(
            spreadsheetId=self.spreadsheetId,
            range=rangeName,
            body=body,
            valueInputOption='USER_ENTERED',
            ).execute()
        return result

    def get_table(self, sheetname, fmt='pandas'):
        values = super().get_data(sheetname)
        if fmt == 'pandas':
            return pd.DataFrame(values[1:], columns=values[0])
        elif fmt == 'numpy':
            return np.array(values)
        elif fmt == 'raw':
            return values
        else:
            raise ValueError('Format not understood')

    def update_tsv_table(self, sheetname, sandbox=True):
        '''Update TSV table from the Google Sheet'''
        from ..filenames import get_sample_table_filename
        fn = get_sample_table_filename(kind=sheetname, sandbox=sandbox)

        table = self.get_table(sheetname=sheetname, fmt='raw')

        table_tsv = '\n'.join(map('\t'.join, table))+'\n'
        with open(fn, 'w') as f:
            f.write(table_tsv)

