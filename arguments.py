import argparse

ARG_TYPES = ['parser', 'train', 'eval']

def get_arguments(arg_type):

    assert arg_type in ARG_TYPES, "Not a valid argument type."

    if arg_type == 'parser':

        parser = argparse.ArgumentParser(description="Corpus Parser")
        parser.add_argument("--dataset-name", required=True, 
                help="The directory name in ./data/ to be parsed.")
        args = parser.parse_args()
    
    elif arg_type == 'train':
        pass

    elif arg_type == 'eval':
        pass

    return args
