__all__ = ['get_mfcc_params', 'get_mel_spectrogram_params']


def set_ft_extr_params_to_args(args):
    if args.ft_extr_type == "mfcc":
        args.n_mfcc_bins, args.ft_extr_params_list = get_mfcc_params(args.params_id)

    elif args.ft_extr_type == "mel_spectrogram":
        args.ft_extr_params_list = get_mel_spectrogram_params(args.params_id)

    elif args.ft_extr_type == "spectrogram":
        args.ft_extr_params_list = get_spectrogram_params(args.params_id)

    elif args.ft_extr_type == "linear_stft":  # n_mels unused
        args.ft_extr_params_list = get_linear_stft_params(args.params_id)

    elif args.ft_extr_type == "raw":
        args.ft_extr_params_list = [(125, 128)]

    # Unused for now
    elif args.ft_extr_type == "lpcc":
        args.ft_extr_params_list = [7, 9, 11, 13, 15]

    elif args.ft_extr_type == "plp":  # Mega slow?
        args.ft_extr_params_list = [10, 15, 20, 25, 30, 35, 40]

    elif args.ft_extr_type == "ngcc":  # n_ceps/order, nfilts TODO in progress, might be dropped
        args.ft_extr_params_list = [(10, 24), (10, 48), (10, 64),
                                    (20, 24), (20, 48), (20, 64),
                                    (30, 24), (30, 48), (30, 64),
                                    (40, 24), (40, 48), (40, 64)]

    else:
        raise NotImplementedError


def get_mfcc_params(params_id):
    """MFCC params, shape (n_mels, win_len), n_mfcc is fixed to 10.
    We choose to fix n_mels to 10, 40, 80 in each runs, as OFA tends to learn only one n_mels configuration when mixing them.
    used:
        - [(40, 40)] n_bin_count=10 #1
        - [(10, 30), (10, 40), (10, 50)], n_bin_count=10 #2
        - [(40, 30), (40, 40), (40, 50)], n_bin_count=10 #3, 40 #5
        - [(80, 30), (80, 40), (80, 50)], n_bin_count=10 #4, 40 #6, 80 #7
        - [(40, 40)] n_bin_count=40 #8
        Experimental:
        - [(40, 40)]
        - [(40, 30), (40, 40), (40, 50),
            (80, 30), (80, 30), (80, 30)] works but 80 is meh
        - [(40, 30), (40, 40), (40, 50)]
        - [(10, 30), (10, 40), (10, 50),
            (20, 30), (20, 40), (20, 50),
            (30, 30), (30, 40), (30, 50),
            (40, 30), (40, 40), (40, 50)]
    """
    if params_id == 1:
        n_mfcc_bins = 10
        ft_extr_params_list = [(40, 40)]
    elif params_id == 2:
        n_mfcc_bins = 10
        ft_extr_params_list = [(10, 30), (10, 40), (10, 50)]
    elif params_id == 3:
        n_mfcc_bins = 10
        ft_extr_params_list = [(40, 30), (40, 40), (40, 50)]
    elif params_id == 4:
        n_mfcc_bins = 10
        ft_extr_params_list = [(80, 30), (80, 40), (80, 50)]
    elif params_id == 5:
        n_mfcc_bins = 40
        ft_extr_params_list = [(40, 30), (40, 40), (40, 50)]
    elif params_id == 6:
        n_mfcc_bins = 40
        ft_extr_params_list = [(80, 30), (80, 40), (80, 50)]
    elif params_id == 7:
        n_mfcc_bins = 80
        ft_extr_params_list = [(80, 30), (80, 40), (80, 50)]
    elif params_id == 8:
        n_mfcc_bins = 40
        ft_extr_params_list = [(40, 40)]
    else:
        raise NotImplementedError("params_id not implemented")

    return n_mfcc_bins, ft_extr_params_list


def get_mel_spectrogram_params(params_id):
    """MelSpectrogram params, shape (n_mels, win_len)
        used:
            - [(10, 20), (10, 25), (10, 30)]
            - [(40, 20), (40, 30), (40, 40)]
            - [(80, 20), (80, 30), (80, 40)]
    """
    if params_id == 1:
        ft_extr_params_list = [(10, 20), (10, 25), (10, 30)]
    elif params_id == 2:
        ft_extr_params_list = [(40, 20), (40, 30), (40, 40)]
    elif params_id == 3:
        ft_extr_params_list = [(80, 20), (80, 30), (80, 40)]
    else:
        raise NotImplementedError("params_id %i is not implemented" % params_id)
    return ft_extr_params_list


def get_spectrogram_params(params_id):
    """Spectrogram params, shape (n_fft, win_len)
        used:
            - [(400, 20), (400, 30), (400, 40)]
            - [(1024, 20), (1024, 30), (1024, 40)]
            - [(2048, 20), (2048, 30), (2048, 40)]
    """
    if params_id == 1:
        ft_extr_params_list = [(400, 20), (400, 30), (400, 40)]
    elif params_id == 3:
        ft_extr_params_list = [(1024, 20), (1024, 30), (1024, 40)]
    elif params_id == 3:
        ft_extr_params_list = [(2048, 20), (2048, 30), (2048, 40)]
    else:
        raise NotImplementedError("params_id %i is not implemented" % params_id)
    return ft_extr_params_list


def get_linear_stft_params(params_id):
    """Linear STFT params, shape (n_fft, win_len)
        used:
            - [(1, 10), (1, 20), (1, 30), (1, 40), (1, 50), (1, 60)]
            - (1024, 40), (1024, 60), (1024, 80),
            (2048, 40), (2048, 60), (2048, 80)]
    """
    if params_id == 1:
        ft_extr_params_list = [(640, 20), (640, 30), (640, 40)]
    elif params_id == 3:
        ft_extr_params_list = [(1024, 20), (1024, 30), (1024, 40)]
    elif params_id == 3:
        ft_extr_params_list = [(2048, 20), (2048, 30), (2048, 40)]
    else:
        raise NotImplementedError("params_id %i is not implemented" % params_id)
    return ft_extr_params_list
