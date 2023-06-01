def evaluate_model():
    if args.model == 'sub':
        model = SubPixelTrainer(args, training_data_loader, testing_data_loader)
        print('sub PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))
    elif args.model == 'srcnn':
        model = SRCNNTrainer(args, training_data_loader, testing_data_loader)
        print('srcnn PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))
    elif args.model == 'vdsr':
        model = VDSRTrainer(args, training_data_loader, testing_data_loader)
        print('vdsr PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))
    elif args.model == 'edsr':
        model = EDSRTrainer(args, training_data_loader, testing_data_loader)
        print('edsr PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))
    elif args.model == 'fsrcnn':
        model = FSRCNNTrainer(args, training_data_loader, testing_data_loader)
        print('fsrcnn PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))
    elif args.model == 'srgan':
        model = SRGANTrainer(args, training_data_loader, testing_data_loader)
        print("srgan PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))
    else:
        raise Exception("the model does not exist")
