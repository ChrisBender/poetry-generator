


if use_pretrained_model:
    print("*** Loading pretrained model ... ***")
    checkpoint = torch.load('./models/lstm.pt')
    net.load_state_dict(checkpoint['model'])
    print("*** Loaded pretrained model. ***")
    for temp in (0.8,):
        for _ in range(3):
            print("*" * 30)
            print("*** Sample for temp = {0}: ***".format(temp))
            print(net.get_sample(max_length=1000, temperature=temp))
    exit(0)


