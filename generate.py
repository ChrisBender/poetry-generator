from arguments import get_arguments
from common import get_model_and_data

import torch

def main():

    args = get_arguments('generate')

    try:
        checkpoint = torch.load("models/" + args.dataset + "/model.pt")
    except OSError:
        print("No model exists at models/" + args.dataset + ". You must train the model"
                + " before generation.")
    net, _ = get_model_and_data(checkpoint['args'])

    net.load_state_dict(checkpoint['model'])

    for _ in range(args.num_samples):
        print(net.get_sample(
            init_string=args.init_string, 
            max_length=args.max_length,
            temperature=args.temperature
        ))

if __name__ == "__main__":
    main()

