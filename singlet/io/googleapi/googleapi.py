# vim: fdm=indent
# author:     Fabio Zanini
# date:       09/01/17
# content:    Access Google Sheets via the REST API.
# Modules
import os
import numpy as np
import pandas as pd



# Classes / functions
class GoogleIOError(IOError):
    pass


class GoogleAPI:
    # If modifying these scopes, delete your previously saved credentials
    # NOTE: readonly for now
    scopes = 'https://www.googleapis.com/auth/spreadsheets.readonly'
    application_name = 'Google Sheet API to singlet'
    MAX_COLUMN = 'AZ'

    def __init__(
            self,
            spreadsheetId,
            spreadsheetname,
            client_id_filename,
            client_secret_filename):
        self.spreadsheetname = spreadsheetname
        self.client_id_filename = client_id_filename
        self.client_secret_filename = client_secret_filename
        self.spreadsheetId = spreadsheetId
        self.set_service()

    def get_credentials(self):
        """Gets valid user credentials from storage.

        If nothing has been stored, or if the stored credentials are invalid,
        the OAuth2 flow is completed to obtain the new credentials.

        Returns:
            Credentials, the obtained credential.
        """
        from oauth2client import client
        from oauth2client import tools
        from oauth2client.file import Storage

        # Cached credentials
        credential_dir = os.path.dirname(self.client_secret_filename)
        if not os.path.exists(credential_dir):
            os.makedirs(credential_dir)

        store = Storage(self.client_secret_filename)
        credentials = store.get()
        if not credentials or credentials.invalid:
            flow = client.flow_from_clientsecrets(self.client_id_filename,
                                                  self.scopes)
            flow.user_agent = self.application_name
            print('Storing credentials to '+self.client_secret_filename)
            credentials = tools.run_flow(flow, store)
            print('Stored credentials into:', self.client_secret_filename)
        return credentials

    def set_service(self):
        import httplib2
        from apiclient import discovery

        credentials = self.get_credentials()
        http = credentials.authorize(httplib2.Http())
        discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?'
                        'version=v4')
        self.service = discovery.build(
                'sheets', 'v4', http=http,
                discoveryServiceUrl=discoveryUrl)

    def get_sheet_shape(self, sheetname):
        '''Get the data range of the sheet'''
        result = self.service.spreadsheets().get(
            spreadsheetId=self.spreadsheetId).execute()
        for sheet in result['sheets']:
            if sheet['properties']['title'] == sheetname:
                break
        else:
            raise GoogleIOError('Sheet not found')

        props = sheet['properties']['gridProperties']
        return [props['rowCount'], props['columnCount']]

    def get_last_column(self, sheetname):
        return self._convert_col_index_to_A1(
            self.get_sheet_shape(sheetname=sheetname)[1] - 1
            )

    def get_header_columns_indices(self, colnames, sheetname):
        '''Get the indices of columns in A1 notation'''
        # Get the column indices
        if self.MAX_COLUMN is None:
            last_col = self.get_last_column(sheetname=sheetname)
        else:
            last_col = self.MAX_COLUMN
        rangeName = sheetname+'!A1:'+last_col
        result = self.service.spreadsheets().values().get(
            spreadsheetId=self.spreadsheetId, range=rangeName).execute()
        header = result.get('values', [])[0]
        if not header:
            raise GoogleIOError('No header data found.')
        icols = {}
        for colname in colnames:
            icols[colname] = self._convert_col_index_to_A1(header.index(colname))
        return icols

    def get_data(self, sheetname, ranges=()):
        '''Get the whole sheet as a pandas dataframe'''
        if not ranges:
            # The max row number can be anything, Google figures it out
            if self.MAX_COLUMN is None:
                last_col = self.get_last_column(sheetname=sheetname)
            else:
                last_col = self.MAX_COLUMN
            ranges = 'A1:'+last_col+'100000'

        if isinstance(ranges, str):
            ranges = [ranges]
            squeeze = True
        else:
            squeeze = False

        ranges = [sheetname+'!'+r for r in ranges]

        # Get the data
        results = []
        for r in ranges:
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheetId, range=r).execute()
            values = result.get('values', [])
            results.append(values)

        if squeeze:
            results = results[0]

        return results

    def set_sheet(self, sheetname, values):
        if self.MAX_COLUMN is None:
            last_col = self.get_last_column(sheetname=sheetname)
        else:
            last_col = self.MAX_COLUMN
        rangeName = sheetname+'!A1:'+last_col+'100000'
        self.service.spreadsheets().values().update(
            spreadsheetId=self.spreadsheetId,
            range=rangeName,
            valueInputOption='RAW',
            body={'values': values},
            ).execute()

    def get_sheetnames(self):
        spreadsheet = self.service.spreadsheets().get(
               spreadsheetId=self.spreadsheetId
                ).execute()
        sheetnames = {s['properties']['sheetId']: s['properties']['title']
                      for s in spreadsheet['sheets']}
        return sheetnames

    def add_sheet(self, sheetname):
        self.service.spreadsheets().batchUpdate(
                spreadsheetId=self.spreadsheetId,
                body={'requests': [
                    {'addSheet': {'properties': {'title': sheetname}}}
                    ]}
                ).execute()

    def delete_sheet(self, sheetname):
        raise NotImplementedError('Function missing ON PURPOSE! Please delete sheets by hand.')

    def get_sheet_id(self, sheetname):
        for sid, sname in self.get_sheetnames().items():
            if sname == sheetname:
                return sid
        raise ValueError('Sheet not found')

    def get_named_ranges(self, sheetname=None, fmt='dict', convert=True):
        sheetnames = self.get_sheetnames()

        response = self.service.spreadsheets().get(
               spreadsheetId=self.spreadsheetId,
               includeGridData=False,
                ).execute()

        # FIXME: Google does not return a sheetID when it is zero (?!)
        for r in response['namedRanges']:
            if 'sheetId' not in r['range']:
                r['range']['sheetId'] = 0

        # restrict to that sheet
        named_ranges = [r for r in response['namedRanges']
                        if (sheetname is None) or
                           (sheetnames[r['range']['sheetId']] == sheetname)]

        if fmt == 'list':
            if convert:
                for r in named_ranges:
                    r['range'] = self.convert_range_json_to_A1(r['range'])
            return named_ranges

        elif fmt == 'dict':
            named_ranges = {r['name']: r['range'] for r in named_ranges}
            if convert:
                named_ranges = {k: self.convert_range_json_to_A1(r)
                                for k, r in named_ranges.items()}
            return named_ranges
        else:
            raise ValueError('Format of named ranges not understood')

    def get_cell_property(self, sheetname, field_string, ranges=()):
        if not ranges:
            ranges = 'A1:'+self.MAX_COLUMN+'100000'

        if isinstance(ranges, str):
            ranges = [ranges]
            squeeze = True
        else:
            squeeze = False

        ranges = ["'"+sheetname+"'!"+r for r in ranges]

        response = self.service.spreadsheets().get(
               spreadsheetId=self.spreadsheetId,
               ranges=ranges,
               includeGridData=True,
                ).execute()

        fields = field_string.split('.')

        data_ranges = []
        for data_raw in response['sheets'][0]['data']:
            data = []
            for row in data_raw['rowData']:
                data.append([])
                if 'values' not in row:
                    continue
                row = row['values']
                for cell in row:
                    datum = None
                    for field in fields:
                        if field not in cell:
                            break
                        cell = cell[field]
                    else:
                        # RGB to HEX
                        if 'color' in fields[-1].lower():
                            cell = tuple([cell['red'], cell['green'], cell['blue']])
                            cell = self.rgb_to_hex(cell)
                        datum = cell
                    data[-1].append(datum)
            data_ranges.append(data)

        if squeeze:
            data_ranges = data_ranges[0]

        return data_ranges

    def set_cell_property(self, sheetname, field_string, data,
                          start_range=(0, 0)):
        sheet_id = self.get_sheet_id(sheetname)

        fields = field_string.split('.')

        # Convert to Google JSON
        data_json = []
        for row in data:
            data_json.append({})
            if not row:
                continue
            data_row = []
            data_json[-1]['values'] = data_row
            for cell in row:
                data_cell = {}
                data_row.append(data_cell)
                if not cell:
                    continue

                for field in fields[:-1]:
                    data_cell[field] = {}
                    data_cell = data_cell[field]
                field = fields[-1]

                # HEX to RGB
                if 'color' in field.lower():
                    cell = self.hex_to_rgb(cell)
                    cell = {'red': cell[0], 'green': cell[1], 'blue': cell[2]}

                data_cell[field] = cell

        # Upload
        self.service.spreadsheets().batchUpdate(
                spreadsheetId=self.spreadsheetId,
                body={'requests': [
                    {'updateCells': {
                        'start': {
                            'sheetId': sheet_id,
                            'rowIndex': start_range[0],
                            'columnIndex': start_range[1],
                        },
                        'fields': field_string,
                        'rows': data_json,
                        }},
                    ]}
                ).execute()

    def set_dimension_size(self, sheetname, dimension, pixel_size,
                           dim_range=(None, None)):
        if dimension not in ('rows', 'columns'):
            raise ValueError('Dimension must be in [\'rows\', \'columns\']')

        sheet_id = self.get_sheet_id(sheetname)

        r = {'sheetId': sheet_id,
             'dimension': dimension.upper(),
             }
        if dim_range[0] is not None:
            r['startIndex'] = dim_range[0]
        if dim_range[1] is not None:
            r['endIndex'] = dim_range[1]

        self.service.spreadsheets().batchUpdate(
               spreadsheetId=self.spreadsheetId,
               body={'requests': [
                   {'updateDimensionProperties': {
                       'range': r,
                       'properties': {'pixelSize': pixel_size},
                       'fields': 'pixelSize',
                       }},
                   ]}
               ).execute()

    def get_backgrounds(self, sheetname):
        return self.get_cell_property(
                sheetname,
                'effectiveFormat.backgroundColor')

    def set_backgrounds(self, sheetname, bkgs,
                        start_range=(0, 0)):
        return self.set_cell_property(
                sheetname,
                'userEnteredFormat.backgroundColor',
                bkgs,
                start_range=start_range)

    def get_text_format(self, sheetname, prop,
                        ranges=()):
        return self.get_cell_property(
                sheetname,
                'effectiveFormat.textFormat.'+prop,
                ranges=ranges)

    def set_text_format(self, sheetname, prop, fmts,
                        start_range=(0, 0)):
        return self.set_cell_property(
                sheetname,
                'userEnteredFormat.textFormat.'+prop,
                fmts,
                start_range=start_range)

    def convert_range_json_to_A1(self, r):
        sheetname = self.get_sheetnames()[r['sheetId']]
        start = self.convert_row_col_to_A1(
                r['startRowIndex'],
                r['startColumnIndex'])
        end = self.convert_row_col_to_A1(
                r['endRowIndex'] - 1,
                r['endColumnIndex'] - 1)
        return "'"+sheetname+"'!"+start+':'+end

    @staticmethod
    def _convert_col_index_to_A1(index):
        def convert(index):
            return chr(ord('A')+index)

        if index < 26:
            return convert(index)
        else:
            return convert(index // 26 - 1)+convert(index % 26)

    @classmethod
    def convert_row_col_to_A1(self, row, col):
        return self._convert_col_index_to_A1(col)+str(row+1)

    @staticmethod
    def convert_row_col_to_A24(row, col):
        return chr(ord('A')+row)+str(col+1)

    @staticmethod
    def convert_A24_to_row_col(well):
        return (ord(well[0]) - ord('A'), int(well[1:]) - 1)

    @staticmethod
    def hex_to_rgb(value):
        """Return (red, green, blue) for the color given as #rrggbb."""
        value = value.lstrip('#')
        lv = len(value)
        return tuple(1.0 / 255 * int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    @staticmethod
    def rgb_to_hex(rgb):
        """Return color as #rrggbb for the given color values."""
        return '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)
