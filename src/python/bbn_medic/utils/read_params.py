# read_param.py - parameter file parsing module.
# This file was imported from pycube, hycube release R2022_07_06_1
import re
import json
import sys
from typing import Set
from enum import Enum


class PARAM_TYPES(Enum):
    # note the _LIST variant means the input string is a space-separated list
    RP_INT = 0         # a scalar integer
    RP_INT_LIST = 1    # a list of integers
    RP_UNS = 2	       # a scalar unsigned int
    RP_UNS_LIST = 3    # a list of unsigned ints
    RP_FLT = 4	       # a scalar float
    RP_FLT_LIST = 5    # a list of floats
    RP_DBL = 6	       # a double
    RP_DBL_LIST = 7    # a list of doubles
    RP_STR = 8	       # a single string
    RP_STR_LIST = 9    # a list of strings
    RP_BOOL = 10       # a scalar boolean
    RP_BOOL_LIST = 11  # a list of booleans
    RP_LIST = 12       # a JSON array, can not be used as command line argument
    RP_UNSPECIFIED = 99  # unspecified type

    @classmethod
    def __to_bool(self, value):
        if isinstance(value, bool):
            return value
        elif isinstance(value, int):
            return bool(value)
        elif isinstance(value, str):
            if value.lower() == 'true' or value == '1':
                return True
            elif value.lower() == 'false' or value == '0':
                return False
        raise Exception('read_param(): invalid boolean ' + value)

    @classmethod
    def _convert_type(self, value, _type):
        if _type == self.RP_INT:     	    # a scalar integer
            return int(value) if value is not None else None
        elif _type == self.RP_INT_LIST:	    # a list of space separated integers
            return [int(x) for x in value.split()]
        elif _type == self.RP_FLT:          # a scalar float
            return float(value) if value is not None else None
        elif _type == self.RP_FLT_LIST:     # a list of space separated floats
            return [float(x) for x in value.split()]
        elif _type == self.RP_STR:          # a single string
            return value
        elif _type == self.RP_STR_LIST:     # a list of space separated strings
            if type(value) == str:
                return value.split()
            elif type(value) == list:
                return value
        elif _type == self.RP_BOOL:          # a scalar boolean
            return self.__to_bool(value)
        elif _type == self.RP_BOOL_LIST:     # a list of booleans
            return [self.__to_bool(x) for x in value.split()]

        elif _type == self.RP_LIST:          # a list
            if not isinstance(value, list):
                raise ValueError(f'{value} is not a JSON array')
            return value
        else:
            raise Exception('read_param(): undefined parameter _type', _type)


class Params(dict):
    """
    Configuration object created from a hash object (from json file)
    """
    def __init__(self, *args, required_params: Set[str]=None, **entries):
        """

        :param args: Arguments without -- prefix are treated as JSON filenames to read parameters from.
                     Arguments with -- prefix are treated as command line arguments.
                     When both formats are given command line arguments take precedence.
                     This means one can override parameters defined in JSON files with command line arguments.

                     The format of command line arguments is --key=value where key and value are strings. If the rvalue
                     string can be parsed as JSON (i.e. a number 123.4, or a list [1,2,3]), it will be treated
                     as such. Otherwise it will be treated as a string. To enforce rvalue to be a string or to include
                     space in the value one needs to quote them from shell: --key=\'123.4\' or --key=\"[1, 2, 3]\".

        :param required_params: set of required parameters
        :param entries: key/value pairs to be merged into the parameter dictionary directly

        Special parameter: list_parameter_values
                     By default full JSON parameters will be listed on stderr.
                     To disable that set one needs to set 'list_parameter_values' to False.
        """
        inner_dict = {}

        json_files = [a for a in args if a[:2] != '--']
        cli_args = [a for a in args if a[:2] == '--']

        # First read parameters from JSON filenames
        for json_file in json_files:
            with open(json_file) as infile:
                config = json.load(infile)
                if isinstance(config, dict):
                    inner_dict.update(config)
                else:
                    raise ValueError('%s is not a dictionary object' % json_file)

        # command line arguments come next, if any
        opt_re = re.compile(r'--(?P<key>[a-zA-Z._-]+)=(?P<value>.+)')
        for arg in cli_args:
            m = opt_re.match(arg)
            if m is None:
                raise ValueError("Invalid command line argument '%s'. Expected format is --key=value." % arg)

            k = m.group('key')
            v = m.group('value')

            # try parsing the string into appropriate type via JSON
            try:
                v = json.loads('{"key":%s}' % v)["key"]
            except json.decoder.JSONDecodeError:
                pass
            if k in inner_dict:
                print("Warning: overriding key '%s' with command line argument '%s'" % (k, arg), file=sys.stderr)
            inner_dict[k] = v

        if entries is not None:
            inner_dict.update(entries)

        # dump params to stderr, stdout maybe used by pipe so let's not use that.
        if 'list_parameter_values' not in inner_dict or inner_dict['list_parameter_values']:
            print('******************** read_params.Params() *******************', file=sys.stderr)
            print(json.dumps(inner_dict, sort_keys=True, indent=4), file=sys.stderr)
            print('', file=sys.stderr)

        self.__dict__.update(inner_dict)
        super().__init__(inner_dict)

        # verify that the params we require to be present are actually there.
        if required_params is not None:
            missing_params = required_params - self.keys()
            if len(missing_params) > 0:
                raise ValueError("Missing required parameters: {}".format(missing_params))

    def get_required_param(self, key: str, param_type: PARAM_TYPES=PARAM_TYPES.RP_UNSPECIFIED):
        if key not in self.keys():
            raise KeyError("Required parameter '" + key + "' could not be found!")
        value = self[key]
        if param_type != PARAM_TYPES.RP_UNSPECIFIED:
            value = PARAM_TYPES._convert_type(value, param_type)
        return value

    def get_optional_param(self, key: str, default_val=None, param_type: PARAM_TYPES=PARAM_TYPES.RP_UNSPECIFIED):
        value = self.get(key, default_val)
        if param_type != PARAM_TYPES.RP_UNSPECIFIED:
            value = PARAM_TYPES._convert_type(value, param_type)
        return value

    # allows this object to act like a dict, see https://stackoverflow.com/questions/4014621/a-python-class-that-acts-like-dict for details
    # note that this is not a complete implementation

    def __getitem__(self, item):
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__[key] = value

    def __str__(self):
        return json.dumps(self.__dict__,
                          sort_keys=True,
                          indent=4,
                          separators=(',', ': '))

    @staticmethod
    def as_bool(par: str) -> bool:
        if par == 1:
            return True
        if par == 0:
            return False
        par = par.lower()
        if par == 'true' or par == '1':
            return True
        elif par == 'false' or par == '0':
            return False
        else:
            raise ValueError("Unable to convert par {} to boolean".format(par))


def read_param(param_file, *args):

    file_params = __read_file_params(param_file)
    program_params = __read_program_params(*args)

    params = {}
    for k in program_params:
        if k in file_params.keys():
            _type = program_params[k]
            v = PARAM_TYPES._convert_type(file_params[k], _type)
            params[k] = v
        else:
            raise Exception("read_param(): parameter string not defined in program", k)

    return Params(**params)


def __read_file_params(param_file):

    RP_PAR_SEP_STR = ":"
    file_params = {}
    pfile = open(param_file)
    for line in pfile:
        line = line.strip()
        if not line or re.match('^;', line):
            continue

        index = line.index(RP_PAR_SEP_STR)
        k = line[:index].strip()
        v = line[index + 1:].strip()
        file_params[k] = v
    return file_params


def __read_program_params(*args):
    program_params = {}
    for i in range(0, len(args), 2):
        program_params[args[i]] = args[i + 1]
    return program_params
