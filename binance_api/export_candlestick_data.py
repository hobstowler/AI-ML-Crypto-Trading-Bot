from binance_api import BinanceAPI
import sys
import getopt

HELP_MESSAGE = "\nUSAGE:"\
    "\npython3 export_candlestick_data.py ticker "\
    "start_date_in_YYYY-MM-DD end_date_in_YYYY-MM-DD "\
    "interval export_filepath"\
    "\n"

def export_candlestick_data(arguments: list):
    binance = BinanceAPI()
    data = binance.get_candlestick_dataframe(
            arguments[0], (arguments[1]), (arguments[2]), 
            int(arguments[3]))
    binance.export_candlestick_dataframe_csv(data, arguments[4])
    
def user_needs_help(arguments, options_dict: dict):
    valid_options = ["-h", "--help"]
    if len(arguments) < 5:
        return True
    if len(arguments) > 5:
        return True
    option_keys = options_dict.keys()
    if "-h" in option_keys:
        return True
    if "--help" in option_keys:
        return True
    for option in option_keys:
        if option not in valid_options:
            return True
    return False

if __name__ == "__main__":
    options, arguments = getopt.getopt(
        sys.argv[1:], shortopts="h", longopts=["help"])
    options_dict = {}
    for opt in options:
        options_dict[opt[0]] = opt[1]
    if user_needs_help(arguments, options_dict):
        print(HELP_MESSAGE)
        exit(0)
    export_candlestick_data(arguments)
    exit(0)
    
