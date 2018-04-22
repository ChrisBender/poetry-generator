import argparse

ARG_TYPES = ['parser', 'train', 'eval']

def get_arguments(arg_type):

    assert arg_type in ARG_TYPES, "Not a valid argument type."

    if arg_type == 'parser':

        parser = argparse.ArgumentParser(description="Corpus Parser")
        parser.add_argument("--dataset", required=True, 
                help="The directory name in ./data/ to be parsed.")

    elif arg_type == 'train':

        parser = argparse.ArgumentParser(description="Training")
        parser.add_argument("--dataset", required=True,
                help="The directory name in ./data/ to use for training.")
        parser.add_argument("--epochs", type=int, 
                default=25, help="Number of epochs to train for.")
        parser.add_argument("--lr", type=float, 
                default=0.001, help="Learning rate.")
        parser.add_argument("--dropout", type=float, 
                default=0.5, help="Dropout.")
        parser.add_argument("--word-hidden-size", dest="word_hidden_size", type=int, 
                default=512, help="Size of the hidden layer of the word LSTM.")
        parser.add_argument("--char-hidden-size", dest="char_hidden_size", type=int, 
                default=256, help="Size of the hidden layer of the char LSTM.")
        parser.add_argument("--word-num-layers", dest="word_num_layers", type=int, 
                default=3, help="Number of layers in the word LSTM.")
        parser.add_argument("--char-num-layers", dest="char_num_layers", type=int, 
                default=3, help="Number of layers in the char LSTM.")
        parser.add_argument("--print-example-every", dest="print_example_every", type=int, 
                default=1, help="Print an example from the model after this number of epochs.")
        parser.add_argument("--training-testing-split", dest="training_testing_split", type=float, 
                default=0.9, help="Fraction of examples which should be used for training.")
        parser.add_argument("--use-glove", dest="use_glove", action="store_true",
                default=False, help="Use GloVe word embeddings.")
        parser.add_argument("--no-cuda", dest="no_cuda", action="store_true",
                default=False, help="Do not use GPU acceleration.")
        parser.add_argument("--verbose", action="store_true",
                default=False, help="Increase verbosity of the output.")

    elif arg_type == 'generate':

        parser=argparse.ArgumentParser(description="Generation")
        parser.add_argument("--num-samples", dest="num_samples", type=int,
                default=1, help="Number of samples to be generated from the model.")

        pass

    return parser.parse_args()
