__all__ = ['get_mfcc_params', 'get_mel_spectrogram_params']


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
        """
    if params_id == 1:
        ft_extr_params_list = [(10, 20), (10, 25), (10, 30)]
    elif params_id == 2:
        ft_extr_params_list = [(40, 20), (40, 30), (40, 40)]
    else:
        raise NotImplementedError("params_id %i is not implemented" % params_id)
    return ft_extr_params_list
